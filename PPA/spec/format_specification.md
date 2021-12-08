# Analyses
All analyses will be described in a single (or multiple, up for debate) json file. The general rules about json apply.

Each analysis will be specified with its name under the top level.

For each analysis, the following fields have to be given:

- `availability`: a field describing when this analysis is available, see Types/case
- `single_x`: like availability. a field describing when a single x can be selected. If a single x cannot be selected, multiple x are assumed. See Types/case.
- `single_y`: like single_x, see Types/case
- `hyperparameters`: a list of hyperparameters. Each list entry is a Types/hyper
- `python_module`: a field describing how the analysis can be started within code, see Types/module

# Types

## case

A case has two fields:
- `always`: if this is `true`, then the condition described by this case is always true
- `cases`: a list of input conditions for which this case is true, see Types/input_condition

## input_codition

An input condition can have either one or two fields. If only one of the fields is set, the other is assumed to be irrelevant (`don't care`)
- `data`: if this is `time`, this case is only valid when the input data is time series data. If this is `batch`, the case is true if the input data is batch-level data.
- `analysis`: if this is `classification`, this case is true if the user wants to perform a classification. If this is `regression`, this case is true if the user wants to perform a regression.

## hyper

Each hyperparameter has three fields (subject to expansion)
- `name`: The common name of the parameter as to be displayed in the UI.
- `internal_name`: The name of the hyperparameter in code (e.g. grid_size if we have a function that uses grid_size).
- `type`: the type of this parameter. These match python types (int, float, str).
- `minimum`: only relevant for numerical types. The minimum value for this parameter (to avoid -5 estimators)

# Further notes for me

we can use importlib to import a module by its name.
we can use `getattr(<object>, <name of attr>)` to get an attribute if we only know its name by string (for us: we can get the function of a module)
we can get a list of all attributes with dir(object/module)