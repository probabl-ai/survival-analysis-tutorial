scripts := $(wildcard notebooks/*.py)
notebooks := $(patsubst %.py,%.ipynb,$(scripts))

.PHONY: all clean notebooks

notebooks: $(notebooks)

%.ipynb: %.py
	jupytext $< --output $@ --execute --set-kernel sa_online_talk

clean:
	rm -f notebooks/*.ipynb
