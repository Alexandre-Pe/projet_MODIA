FROM pytorch/pytorch:latest
WORKDIR /workspace
ADD requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt
ADD *.py weight.pth /workspace/
ADD data/interactions*.csv pipeline_part2.pkl /workspace/data/
ADD test_script.csv /workspace/
ENTRYPOINT [ "python3" ]
