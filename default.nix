{ buildPythonPackage, notebook, matplotlib, numpy }:

buildPythonPackage {
  name = "arc2020";

  src = ./.;

  propagatedBuildInputs = [ matplotlib numpy ];
  nativeBuildInputs = [ notebook ];
}
