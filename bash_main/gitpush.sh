#! /bin/bash

# git add -- . ':!abatement/data_2tech' ':!job-outs' ':!bash' ':!bash_main'
git add -- . ':!abatement/data_2tech' ':!job-outs' ':!job-outs_old' ':!bash' 
git commit -m 'new commit'
git push
