# Python Scrcpy Client
<p>
    <a href="https://pypi.org/project/scrcpy-client/" target="_blank">
        <img src="https://img.shields.io/pypi/v/scrcpy-client" />
    </a>
    <a href="https://github.com/leng-yue/py-scrcpy-client/actions/workflows/ci.yml" target="_blank">
        <img src="https://img.shields.io/github/workflow/status/leng-yue/py-scrcpy-client/CI" />
    </a>
    <a href="https://app.codecov.io/gh/leng-yue/py-scrcpy-client" target="_blank">
        <img src="https://img.shields.io/codecov/c/github/leng-yue/py-scrcpy-client" />
    </a>
    <img src="https://img.shields.io/github/license/leng-yue/py-scrcpy-client" />
    <a href="https://github.com/Genymobile/scrcpy/tree/v1.18" target="_blank">
        <img src="https://img.shields.io/badge/scrcpy-v1.18-violet" />
    </a>
</p>

This package allows you to view and control android device in realtime.

![demo gif](https://raw.githubusercontent.com/leng-yue/py-scrcpy-client/main/demo.gif)

## How to use
To begin with, you need to install this package via pip:
```shell
pip install scrcpy-client[ui]
```
Then, you can start `py-scrcpy` to view the demo:

Note: you can ignore `[ui]` if you don't want to view the demo ui

## Document
Please check `scrcpy_ui/main.py` and `scrcpy/control.py`, document is not ready now.

## Contribution & Development
Already implemented all functions in scrcpy server 1.18.  
Please check scrcpy server 1.18 source code: [Link](https://github.com/Genymobile/scrcpy/tree/v1.18/server)

## Reference & Appreciation
- Core: [scrcpy](https://github.com/Genymobile/scrcpy)
- Idea: [py-android-viewer](https://github.com/razumeiko/py-android-viewer)
- CI: [index.py](https://github.com/index-py/index.py)
