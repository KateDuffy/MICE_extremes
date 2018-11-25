#!/bin/bash

rsync -avz --progress ~/Documents/dev_codes/ sdslab:~/Documents/dev_codes/

rsync -avz --progress sdslab:~/Documents/dev_codes/ ~/Documents/dev_codes/


