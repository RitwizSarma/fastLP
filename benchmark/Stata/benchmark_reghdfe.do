version 17
clear all
set more off

* Direct HDFE comparison with fastLP. Each timing covers all horizons.
args requested_dataset
local dataset "`requested_dataset'"
if "`dataset'" == "" local dataset "n5000_t40"
local repetitions = 200
local benchmark_dir ".."
local input_file "`benchmark_dir'/data/`dataset'.csv"
local output_file "`benchmark_dir'/results_reghdfe_`dataset'.csv"
local estimates_output_file "`benchmark_dir'/estimates_reghdfe_`dataset'.csv"

local horizon = .
if "`dataset'" == "n5000_t40"  local horizon = 12
if "`dataset'" == "n10000_t40" local horizon = 12
if missing(`horizon') {
    display as error "Unknown dataset: `dataset'"
    exit 198
}

capture which reghdfe
if _rc {
    display as error "reghdfe is not installed. Run: ssc install reghdfe"
    exit 199
}

import delimited using "`input_file'", clear varnames(1)
sort unit time
xtset unit time
local n_rows = _N

tempname results_handle
tempfile results_file
postfile `results_handle' str32 dataset int repetition double wall_seconds user_seconds system_seconds using `results_file', replace
tempname estimates_handle
tempfile estimates_file
postfile `estimates_handle' str32 dataset int horizon double estimate_shock estimate_control using `estimates_file', replace
matrix coefficient_estimates = J(`horizon' + 1, 2, .)
forvalues rep = 1/`repetitions' {
    timer clear 1
    timer on 1
    forvalues h = 0/`horizon' {
        quietly reghdfe F`h'.outcome shock control, absorb(unit time) vce(cluster unit)
        if `rep' == 1 {
            matrix coefficient_estimates[`h' + 1, 1] = _b[shock]
            matrix coefficient_estimates[`h' + 1, 2] = _b[control]
        }
    }
    timer off 1
    quietly timer list 1
    post `results_handle' ("`dataset'") (`rep') (r(t1)) (.) (.)
    if `rep' == 1 {
        forvalues estimate_h = 0/`horizon' {
            post `estimates_handle' ("`dataset'") (`estimate_h') ///
                (coefficient_estimates[`estimate_h' + 1, 1]) ///
                (coefficient_estimates[`estimate_h' + 1, 2])
        }
    }
}
postclose `results_handle'
postclose `estimates_handle'

preserve
use `results_file', clear
order dataset repetition wall_seconds user_seconds system_seconds
export delimited using "`output_file'", replace
quietly summarize wall_seconds, detail
display as text "Dataset: `dataset'"
display as text "Rows: " %12.0fc `n_rows'
display as text "Repetitions: `repetitions'"
display as text "Wall-clock timing summary (seconds):"
display as text "  mean   " %12.6f r(mean)
display as text "  sd     " %12.6f r(sd)
display as text "  median " %12.6f r(p50)
display as text "  Q1     " %12.6f r(p25)
display as text "  Q3     " %12.6f r(p75)
display as text "  min    " %12.6f r(min)
display as text "  max    " %12.6f r(max)
display as text "Results written to: `output_file'"
restore
preserve
use `estimates_file', clear
order dataset horizon estimate_shock estimate_control
export delimited using "`estimates_output_file'", replace
display as text "Estimates written to: `estimates_output_file'"
restore
