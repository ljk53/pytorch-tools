# Quick Start

## Install Valgrind

### On CentOS
```bash
sudo dnf install valgrind valgrind-devel
```

### On Mac
Follow the instructions at: https://stackoverflow.com/questions/58360093/how-to-install-valgrind-on-macos-catalina-10-15-with-homebrew
```bash
brew tap LouisBrunner/valgrind
brew install --HEAD LouisBrunner/valgrind/valgrind
```

## Build & Run

### Build libtorch locally
This script will checkout pytorch source code and build it from scratch.
You can use this workflow to modify pytorch code locally and run the benchmark against the modified version.
```bash
LIBTORCH=local ./callgrind.sh
```
If you already have a working copy of PyTorch repo, you could use it instead:
```bash
PYTORCH_ROOT=<your_pytorch_src_repo> LIBTORCH=local ./callgrind.sh
```

### Download prebuilt libtorch from the official website
This is convenient for quick experiment if you don't need modify/build PyTorch code locally. Note that the prebuilt library doesn't contain debug symbols so you won't be able to see source code level annotation in the benchmark results.
```bash
# On Linux
./callgrind.sh

# On Mac
LIBTORCH=macos ./callgrind.sh
```

## Show Annotated Result

Use command line flags: https://www.valgrind.org/docs/manual/cl-manual.html

```bash
callgrind_annotate \
  --auto=yes \
  --inclusive=yes \
  --tree=both \
  --show-percs=yes \
  --context=16 \
  --include=pytorch \
  callgrind.out.txt
```

## Visualize Result

### Install Kcachegrind
```bash
# On CentOS
sudo dnf install kcachegrind

# On Mac
brew install qcachegrind
```

### Visualize callgraph / source code / assembly instructions and etc.

![Kcachegrind Screenshot](callgrind_demo.png?raw=true)

#### Call Graph / Callee Map

![Kcachegrind Screenshot](callgrind_demo_1.gif?raw=true)

#### Inclusive Cost / Self Cost / Call Count

![Kcachegrind Screenshot](callgrind_demo_2.gif?raw=true)

#### Source Code / Machine Code

![Kcachegrind Screenshot](callgrind_demo_3.gif?raw=true)

## Appendix - Setup VNC Server on CentOS Server

### Setup VNC server and desktop environment

Install the following software on your server:
```bash
sudo dnf install tigervnc-server tigervnc-server-module
sudo dnf groupinstall "Server with GUI" 
```

Setup VNC password with:
```bash
vncpasswd
```

### Start VNC server

#### This can be done by simply running the vncserver.
```bash
/usr/bin/vncserver -depth 24 -geometry 1920x1080 :0
```

#### Alternatively, we could use systemctl service.

First, create a service file at: `/etc/systemd/system/vncserver@.service`
```
[Unit]
Description=Remote desktop service (VNC)
After=syslog.target network.target

[Service]
Type=forking
User=YOUR_USER_ID
Group=users
WorkingDirectory=/home/YOUR_USER_ID

PIDFile=/home/YOUR_USER_ID/.vnc/%H%i.pid
ExecStartPre=-/usr/bin/vncserver -kill %i > /dev/null 2>&1
ExecStart=/usr/bin/vncserver -depth 24 -geometry 1920x1080 %i
ExecStop=/usr/bin/vncserver -kill %i > /dev/null 2>&1

[Install]
WantedBy=multi-user.target
```

Then, try to start it.
```bash
# reload the daemon to read the service config
sudo systemctl daemon-reload

# start the service
sudo systemctl restart vncserver@:0.service

# check the status of the service
sudo systemctl status vncserver@:0.service
```

You can check the log at:
```
tail ~/.vnc/YOUR_HOST_NAME\:0.log
```

You should be able to see the port 5900 is being listened:
```
netstat -tunlp | grep 5900
```

### 

Setup SSH tunnel on your laptop:
```
ssh -L 5900:127.0.0.1:5900 YOUR_SERVER_HOST_NAME
```

Then you can launch VNC viewer on your laptop and connect to `127.0.0.1:5900`.

### Troubleshooting

### D-Bus connection refused

You might see the following error in the log:
```
dbus-update-activation-environment: error: unable to connect to D-Bus: Failed to connect to socket /tmp/...: Connection refused  
```

This can be worked around by adding `dbus-launch gnome-session` to `~/.vnc/xstartup`:
```bash
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS

# >>> add this line to fix dbus connection error <<<
dbus-launch gnome-session

/etc/X11/xinit/xinitrc
```

