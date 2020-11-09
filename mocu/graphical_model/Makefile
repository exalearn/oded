CMD = python
FLAGS = install
DOC = setup.py

all : setup

setup : $(DOC)
	$(CMD) $(DOC) install

test :
	$(CMD) $(DOC) test

script_sde :
	$(CMD) -c "import mocu.scripts.example_sde; mocu.scripts.example_sde.main()"

visualize_script_sde :
	$(CMD) -c "import mocu.scripts.visualizetoysystem; mocu.scripts.visualizetoysystem.main()"

profile_script_sde :
	python -m cProfile -o profile.out mocu/scripts/example_profile.py

script_scaling :
	$(CMD) -c "import mocu.scripts.example_scaling; mocu.scripts.example_scaling.main()"

profile_script_dag :
	python -m cProfile -o profile.out mocu/scripts/example_profile_dag.py

test_dehghannasiri :
	$(CMD) -c "import mocu.verification_tests.test_dehghannasiri; mocu.verification_tests.test_dehghannasiri.main()"

script_ode :
	$(CMD) -c "import mocu.scripts.example_ode; mocu.scripts.example_ode.select_experiment()"

script_linsys :
	$(CMD) -c "import mocu.scripts.example_linearsys; mocu.scripts.example_linearsys.main()"


clean :
	rm -rf build/ dist/ *egg-info

