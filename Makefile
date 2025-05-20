.PHONY: clean notebooks

notebooks:
	jupytext --to notebook --from py:percent tutorial.py --execute

clean:
	rm -f *.ipynb
