{ buildPythonPackage, notebook }:

buildPythonPackage {
  name = "arc2020";

  src = ./.;

  propagatedBuildInputs = [ ];
  nativeBuildInputs = [ notebook ];
}
