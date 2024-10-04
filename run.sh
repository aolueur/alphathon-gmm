python -m gens.gen_sp500
python -m gens.gen_spsector_returns
python -m gens.gen_data
python gmm_run.py
python data_merge.py
python -m gens.gen_summary
python -m gens.gen_weights
python strategy.py
