Steps to make your siglab_py package available via pip install:

1. Add "pyproject.toml" and "setup.cfg"
	\ siglib
		\ siglib_py
			├── \ exchanges
					any_exchange.py
			├── \ ordergateway
			├── \ util
			├── \ tests
			├── README.md
		├── pyproject.toml
		├── setup.cfg
		
2. Build, after you run below commands you should have "dist" folder under project root (i.e. siglib, not siglib_py).
	pip install build
	python -m build

3. Push to PyPI
	python -mpip install --no-deps twine

	python -m twine upload dist/*
	Uploading distributions to https://upload.pypi.org/legacy/
	Enter your API token:	<-- Register here: https://pypi.org/account/register/
	
	Uploading distributions to https://upload.pypi.org/legacy/
	Enter your API token:
	Uploading siglib_py-0.1.0-py3-none-any.whl
	100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.0/41.0 kB • 00:00 • ?
	Uploading siglib_py-0.1.0.tar.gz
	100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.0/40.0 kB • 00:00 • ?

	View at:
	https://pypi.org/project/siglib-py/0.1.0/