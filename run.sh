# git pull

# Delete all `.Ds_Store`
find . -name ".DS_Store" -delete

# De;ete all `.ipynb_checkpoints` folders
find . -name .ipynb_checkpoints -type d -exec rm -rf {} \;

# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute --inplace *.ipynb
# jupyter nbconvert --to script *.ipynb
# ls *.py|xargs -n 1 -P 4 ipython

ipython Bogota_Data.py

ipython Bogota_R_t.py

black *.py

find . -type f -name osb\* -exec rm {} \;

git add --all && git commit -m "Update" && git push
