## Quick Start

1. Run the script locally to install dependencies and test out the flow:
```
codebrowser/build.sh
```

2. If everything works, add it to crontab:
```
$ crontab -l
SHELL=/bin/bash
BASH_ENV=~/.bashrc_conda
MAILTO=<your email>

# m h  dom mon dow   command
0 5 * * * <src path>/pytorch-tests/codebrowser/build.sh
```

3. It generates the website at `${HOME}/www`. If it's your github.io repo, then the script will also run `git push` to publish the change after each run.
