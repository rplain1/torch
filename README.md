
<!-- README.md is generated from README.Rmd. Please edit that file -->

# torch <a href='https://torch.mlverse.org'><img src='man/figures/torch.png' align="right" height="139" /></a>

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
![R build
status](https://github.com/mlverse/torch/workflows/Test/badge.svg)
[![CRAN
status](https://www.r-pkg.org/badges/version/torch)](https://CRAN.R-project.org/package=torch)
[![](https://cranlogs.r-pkg.org/badges/torch)](https://cran.r-project.org/package=torch)

## Installation

torch can be installed from CRAN with:

``` r
install.packages("torch")
```

You can also install the development version with:

``` r
remotes::install_github("mlverse/torch")
```

At the first package load additional software will be installed.

## Examples

You can create torch tensors from R objects with the `torch_tensor`
function and convert them back to R objects with `as_array`.

``` r
library(torch)
x <- array(runif(8), dim = c(2, 2, 2))
y <- torch_tensor(x, dtype = torch_float64())
y
#> torch_tensor
#> (1,.,.) = 
#>   0.1512  0.8540
#>   0.3250  0.3191
#> 
#> (2,.,.) = 
#>   0.8256  0.1999
#>   0.1343  0.4721
#> [ CPUDoubleType{2,2,2} ]
identical(x, as_array(y))
#> [1] TRUE
```

### Simple Autograd Example

In the following snippet we let torch, using the autograd feature,
calculate the derivatives:

``` r
x <- torch_tensor(1, requires_grad = TRUE)
w <- torch_tensor(2, requires_grad = TRUE)
b <- torch_tensor(3, requires_grad = TRUE)
y <- w * x + b
y$backward()
x$grad
#> torch_tensor
#>  2
#> [ CPUFloatType{1} ]
w$grad
#> torch_tensor
#>  1
#> [ CPUFloatType{1} ]
b$grad
#> torch_tensor
#>  1
#> [ CPUFloatType{1} ]
```

## Contributing

No matter your current skills it’s possible to contribute to `torch`
development. See the contributing guide for more information.
