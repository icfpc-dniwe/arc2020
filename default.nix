{ buildPythonPackage, notebook, matplotlib, numpy, numba }:

buildPythonPackage {
  name = "arc2020";

  src = ./.;

  propagatedBuildInputs = [ matplotlib numpy numba ];
  nativeBuildInputs = [ notebook ];
}
