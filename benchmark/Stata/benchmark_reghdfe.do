version 17
clear all
set more off

* Direct, common-sample HDFE comparison with fastLP.
local dataset "small_balanced"
local benchmark_dir ".."
local input_file "`benchmark_dir'/data/`dataset'.csv"

local horizon = .
if "`dataset'" == "small_balanced"   local horizon = 12
if "`dataset'" == "small_unbalanced" local horizon = 12
if "`dataset'" == "large_balanced"   local horizon = 2
if "`dataset'" == "large_unbalanced" local horizon = 2
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

* Include common-sample construction in the timed region, matching fastLP.fit.
timer clear 1
timer on 1
generate byte common_sample = 1
forvalues h = 0/`horizon' {
    replace common_sample = 0 if missing(F`h'.outcome)
}
forvalues h = 0/`horizon' {
    quietly reghdfe F`h'.outcome shock control if common_sample, ///
        absorb(unit time) vce(cluster unit)
}
timer off 1
timer list 1

display as text "reghdfe common-sample LP calculation completed for `dataset'."
