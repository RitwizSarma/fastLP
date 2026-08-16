version 17
clear all
set more off

* Optional Stata locproj implementation using the same repeated timing design.
args requested_dataset
local dataset "`requested_dataset'"
if "`dataset'" == "" local dataset "n5000_t40"
local repetitions = 200
local benchmark_dir ".."
local input_file "`benchmark_dir'/data/`dataset'.csv"
local output_file "`benchmark_dir'/results_locproj_`dataset'.csv"
local estimates_output_file "`benchmark_dir'/estimates_locproj_`dataset'.csv"

local horizon = .
if "`dataset'" == "n5000_t40"  local horizon = 12
if "`dataset'" == "n10000_t40" local horizon = 12
if missing(`horizon') {
    display as error "Unknown dataset: `dataset'"
    exit 198
}

capture which locproj
if _rc {
    display as error "locproj is not installed. Install it (for example, ssc install locproj) and rerun."
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
forvalues rep = 1/`repetitions' {
    timer clear 1
    timer on 1
    quietly locproj outcome, shock(shock) controls(control i.time) hor(`horizon') fe cluster(unit) nograph
    timer off 1
    quietly timer list 1
    post `results_handle' ("`dataset'") (`rep') (r(t1)) (.) (.)
    if `rep' == 1 {
        * Re-run horizon by horizon outside the timed region so e(b) can be
        * captured for every horizon without affecting timing results.
        forvalues estimate_h = 0/`horizon' {
            quietly locproj outcome, shock(shock) controls(control i.time) hor(`estimate_h') fe cluster(unit) nograph
            post `estimates_handle' ("`dataset'") (`estimate_h') (_b[shock]) (_b[control])
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
