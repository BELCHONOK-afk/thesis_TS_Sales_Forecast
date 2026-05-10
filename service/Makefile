up:
	docker compose up --build

down:
	docker compose down

restart:
	docker compose down
	docker compose up --build

api:
	python -m uvicorn app.main:app --reload

dashboard:
	streamlit run dashboard/streamlit_app.py

train-ml:
	python -m training.train_ml_models

train-ml-docker:
	docker compose run --rm training