version 17
clear all
set more off

* Choose one: small_balanced, large_balanced, small_unbalanced, large_unbalanced.
local dataset "small_balanced"

* This do-file is intended to be run from benchmark/Stata.
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

capture which locproj
if _rc {
    display as error "locproj is not installed. Install it (for example, ssc install locproj) and rerun."
    exit 199
}

import delimited using "`input_file'", clear varnames(1)
sort unit time
xtset unit time

* i.time supplies time fixed effects; fe supplies unit fixed effects. The
* cluster() option clusters standard errors at the unit level.
timer clear 1
timer on 1
locproj outcome, shock(shock) controls(control i.time) hor(`horizon') fe cluster(unit) nograph
timer off 1
timer list 1

display as text "locproj LP calculation completed for `dataset'."
