<coding-rules>

1. 테스트가 용이한 구조
2. 수정이 용이한 구조
3. 파일 하나가 쓸데없이 크지 않은 구조

</coding-rules>

<dev-environment>

1. gpu 없는 사무용 노트북
2. python은 venv

</dev-environment>

<train-environment>

1. Runpod 의 rtx 3090 24gb(with AMD EPYC 7C13)
2. Pod template : runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 (+Jupyter)

</train-environment>

<current-notice>

1. `docs` 폴더 내의 파일들은 최대한 무시한다 (outdated)

</current-notice>