# Relational Causal Discovery with $\sigma$-separation (sRCD)

This is a modified version of the causal structure learning algorithm called Relational Causal Discovery (RCD). This version facilitates the experiments provided in the paper titled *Learning Relational Causal Models with Cycles through Relational Acyclification*. The original RCD implementation was released [here](https://kdl-umass.github.io/relational_causal_discovery.html) which is no longer available. A working copy of the original software can be found [here](https://github.com/ragib06/RCD). 

 
## Installations

  ```conda env create -f conda_venv_rcd.yml```
		
  It will create a conda venv named "rcd" which will contain all the required packages. You need to activate the venv before using it:
  
  ```source activate rcd```
  
  
## Running examples:

- Run Oracle example:

	```python src/runOracleRCD.py```

- Run Database Example: 
	- Configure Postgres database and load the dump file **src/rcd-test-data.sql**
	
	- Then run the following: 

		```python src/runDatabaseRCD.py```
  
----------
**References**

Ragib Ahsan, David Arbour, Elena Zheleva (2022) Relational Causal Models with Cycle: Representation and Reasoning. In *Proceedings of 1st Conference on Causal Learning and Representation (CLeaR 2016)*.

Sanghack Lee and Vasant Honavar (2016) On Learning Causal Models for Relational Data.  In *Proceedings of Thirtieth AAAI Conference on Artificial Intelligence (AAAI 2016)*.

Marc Maier, Katerina Marazopoulou, David Arbour, and David Jensen (2013) A sound and complete algorithm for learning causal models from relational data. In *Proceedings of the Twenty-Ninth UAI Conference on Uncertainty in Artificial Intelligence, (UAI-2013)*
