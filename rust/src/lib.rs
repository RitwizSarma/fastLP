//! Optional fixed-effect demeaning backend for fastLP.
//!
//! Adapted from the MIT-licensed `igerber/diff-diff` MAP demeaning kernel.

use numpy::ndarray::ShapeBuilder;
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
struct HdfePlan {
    n: usize,
    code_columns: Vec<Vec<usize>>,
    group_counts: Vec<usize>,
    group_sizes: Vec<Vec<usize>>,
}

#[pymethods]
impl HdfePlan {
    #[new]
    fn new(codes: PyReadonlyArray2<'_, i64>, group_counts: Vec<usize>) -> PyResult<Self> {
        let codes = codes.as_array();
        if codes.ncols() != group_counts.len() {
            return Err(PyValueError::new_err("incompatible fixed-effect inputs"));
        }
        let n = codes.nrows();
        let mut code_columns = Vec::with_capacity(codes.ncols());
        let mut group_sizes = Vec::with_capacity(codes.ncols());
        for dimension in 0..codes.ncols() {
            let mut column = Vec::with_capacity(n);
            let mut sizes = vec![0usize; group_counts[dimension]];
            for row in 0..n {
                let code = codes[(row, dimension)];
                if code < 0 || code as usize >= group_counts[dimension] {
                    return Err(PyValueError::new_err(
                        "fixed-effect code is outside its declared range",
                    ));
                }
                column.push(code as usize);
                sizes[code as usize] += 1;
            }
            code_columns.push(column);
            group_sizes.push(sizes);
        }
        Ok(Self {
            n,
            code_columns,
            group_counts,
            group_sizes,
        })
    }

    /// Apply symmetric alternating projections. Immutable FE topology and
    /// denominators are retained by the plan; per-column work buffers are
    /// allocated once and reused across iterations.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>)> {
        let x = x.as_array();
        if x.nrows() != self.n {
            return Err(PyValueError::new_err("incompatible demeaning input"));
        }
        let n_columns = x.ncols();
        let mut columns: Vec<Vec<f64>> = (0..n_columns)
            .map(|j| (0..self.n).map(|i| x[(i, j)]).collect())
            .collect();
        let sweep_dimensions: Vec<usize> = (0..self.code_columns.len())
            .chain((0..self.code_columns.len().saturating_sub(1)).rev())
            .collect();
        let iterations: Vec<i64> = py.allow_threads(|| {
            columns
                .par_iter_mut()
                .map(|column| {
                    let mut previous = vec![0.0; self.n];
                    let mut sums: Vec<Vec<f64>> = self
                        .group_counts
                        .iter()
                        .map(|&count| vec![0.0; count])
                        .collect();
                    for iteration in 1..=max_iter {
                        previous.copy_from_slice(column);
                        for &dimension in &sweep_dimensions {
                            sums[dimension].fill(0.0);
                            for row in 0..self.n {
                                sums[dimension][self.code_columns[dimension][row]] += column[row];
                            }
                            for row in 0..self.n {
                                let group = self.code_columns[dimension][row];
                                column[row] -= sums[dimension][group]
                                    / self.group_sizes[dimension][group] as f64;
                            }
                        }
                        if column
                            .iter()
                            .zip(previous.iter())
                            .all(|(a, b)| (a - b).abs() < tol)
                        {
                            return iteration as i64;
                        }
                    }
                    -1
                })
                .collect()
        });
        let flat: Vec<f64> = columns.into_iter().flatten().collect();
        let output = numpy::ndarray::Array2::from_shape_vec((self.n, n_columns).f(), flat)
            .map_err(|_| PyValueError::new_err("could not shape output"))?;
        Ok((
            PyArray2::from_owned_array_bound(py, output),
            PyArray1::from_vec_bound(py, iterations),
        ))
    }
}

#[pyfunction]
fn demean_map<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    codes: PyReadonlyArray2<'py, i64>,
    group_counts: Vec<usize>,
    tol: f64,
    max_iter: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>)> {
    let plan = HdfePlan::new(codes, group_counts)?;
    plan.transform(py, x, tol, max_iter)
}

/// Accumulate one cluster-score meat matrix per residual column.
///
/// The Python covariance layer precomputes each one-way or intersection
/// cluster code.  Keeping this array-only kernel independent of pandas lets it
/// accelerate every term in a multiway inclusion--exclusion covariance without
/// changing the shared LP design or materializing observation-score tensors.
#[pyfunction]
fn cluster_meat<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    residuals: PyReadonlyArray2<'py, f64>,
    codes: PyReadonlyArray1<'py, i64>,
    n_clusters: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let x = x.as_array();
    let residuals = residuals.as_array();
    let codes = codes.as_array();
    if x.nrows() != residuals.nrows() || x.nrows() != codes.len() || n_clusters < 2 {
        return Err(PyValueError::new_err(
            "incompatible clustered covariance inputs",
        ));
    }
    if codes
        .iter()
        .any(|&code| code < 0 || code as usize >= n_clusters)
    {
        return Err(PyValueError::new_err(
            "cluster codes are outside the declared range",
        ));
    }
    let n = x.nrows();
    let k = x.ncols();
    let h = residuals.ncols();
    let output = py.allow_threads(|| {
        let mut output = vec![0.0; h * k * k];
        // One horizon at a time bounds temporary memory at G * K, rather than
        // G * K * H.  The outer Python layer batches horizons if necessary.
        for horizon in 0..h {
            let mut scores = vec![0.0; n_clusters * k];
            for row in 0..n {
                let offset = codes[row] as usize * k;
                let residual = residuals[(row, horizon)];
                for column in 0..k {
                    scores[offset + column] += x[(row, column)] * residual;
                }
            }
            let out_offset = horizon * k * k;
            for group in 0..n_clusters {
                let score_offset = group * k;
                for left in 0..k {
                    let left_score = scores[score_offset + left];
                    for right in 0..k {
                        output[out_offset + left * k + right] +=
                            left_score * scores[score_offset + right];
                    }
                }
            }
        }
        output
    });
    let output = numpy::ndarray::Array3::from_shape_vec((h, k, k), output)
        .map_err(|_| PyValueError::new_err("could not shape clustered covariance output"))?;
    Ok(PyArray3::from_owned_array_bound(py, output))
}

#[pymodule]
fn _fastlp_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<HdfePlan>()?;
    module.add_function(wrap_pyfunction!(demean_map, module)?)?;
    module.add_function(wrap_pyfunction!(cluster_meat, module)?)?;
    Ok(())
}
