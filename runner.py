import sys
import os
from base64 import b64decode
from tempfile import NamedTemporaryFile

if __name__ == "__main__":
    ZIPFILE = b64decode("REPLACE_WITH_ARCHIVE")

    with NamedTemporaryFile(prefix="arc2020", suffix=".pyz", delete=False) as tmp:
        try:
            tmp.write(ZIPFILE)
            package_path = tmp.name
        except:
            os.remove(tmp.name)
            raise

    try:
        sys.path.insert(0, package_path)
        import arc2020
        arc2020.main()
    finally:
        try:
            sys.path.remove(package_path)
        except ValueError:
            pass
        os.remove(package_path)
