# Sample app for analyzing facial emotion using Affectiva's ICS SDK (C++ native library for Aarch64)

### frame-detector-video-demo

This sample demonstrates use of the SyncFrameDetector class, getting its input from a video file. It analyzes received frames and displays the results on screen.

After building, run the command `./frame-detector-video-demo --help` for information on its command line options.

---

## Dependencies

#### Affectiva Vision library

The Affectiva ICS SDK is available upon request. To get access, please [contact us](https://auto.affectiva.com).


#### Linux (x86_64, aarch64)
Branches like this one that have a name that includes "qc", are Android-specific (ARM64) and aarch64-linux-gnu based. For samples that work on Linux, switch to the branch whose name that matches the release you're using.
##Instructions

### To install, build and run docker 

If you're new to Docker [click here](https://docs.docker.com/get-started) to learn more about it and install it on your host machine. 

After installing Docker, use the Dockerfile located in the top-level directory of this repo ([here](../Dockerfile.aarch64)) to build the docker image.

###### Building docker image:
Setup the environment variable  
 `$ export GIT_COMMIT="$(git rev-parse HEAD)"`
 
Build the docker image on the host machine, this docker image will also contain the sample app  
 `$ docker build -f Dockerfile.aarch64 --build-arg AFFECTIVA_ICS_SDK_URL=<download URL> --build-arg BRANCH=<branch of this repo> --tag=affectiva:ics-sdk-$GIT_COMMIT .`

Above command will generate the artifact (sample app), which we'll push it to the target device (Qualcomm SA-8155P development board)

###### Run this container interactively (optional): 
 `$ docker run -it --rm affectiva:ics-sdk-$GIT_COMMIT`


### To copy artifact from docker image and push it to the target device
Create and start a docker container.  
`$ docker create -ti --name affectiva-ics-sdk affectiva:ics-sdk-$GIT_COMMIT bash` 

Copy the generated artifact (which was created during docker build) to specified path.  
`$ docker cp affectiva-ics-sdk:/opt/testapp-artifact.tar.gz <target/path/to/copy/artifact>` 

Delete the created container.  
`$ docker rm -f affectiva-ics-sdk` 

Now, setup your adb device where you'll test the app, for more info on adb ([click here](https://developer.android.com/studio/command-line/adb )) 

It is recommended to run all the below commands after running `sudo -s` on the terminal.

Restart adb connection with root privileges (use this before starting any push operations).  
`$ adb root` 

Confirm that your host computer is connected to the target device:  
`$ adb devices`

After establishing successful connection to the target device (can be verified by `$ adb devices`), push the artifact to it by using following command

`$ adb push <path/where/the/artifact/resides> <path/on/the/target/device>`

Usually it's recommended to push the application to `/data/loca/tmp/` directory on the target device.
One can either push the .tar.gz file directly and then extract it on the target device or extract the file locally and then push it to the target device. 
The former method is faster and easier

Copy the artifact, and use same approach to copy any other files to the target device.   
`$ adb push testapp-artifact.tar.gz /data/loca/tmp/` 

Note:: The above step might take up to 100 secs, depends on the transfer speed.

### To setup and run frame-detector-video-demo on the target device

Enter into android's shell env.  
`$ adb shell`  

Go into the tmp directory where the artifact was pushed or copied.  
`$ cd /data/local/tmp/` 

Create a testapp directory.  
`$ mkdir affectiva-cpp-testapp`

Extract artifact into the current directory.  
`$ tar -xf testapp-artifact.tar.gz -C affectiva-cpp-testapp`

Set $LD_LIBRARY_PATH env variable.  
`$ export LD_LIBRARY_PATH=/data/local/tmp/affectiva-cpp-testapp/lib/arm64-v8a`

Set data directory $AFFECTIVA_VISION_DATA_DIR env variable.  
`$ export AFFECTIVA_VISION_DATA_DIR=/data/local/tmp/affectiva-cpp-testapp/data`

Change the directory where the sample app is present.  
`$ cd affectiva-cpp-testapp/bin`

run the command `./frame-detector-video-demo --help` for information on how to run it. Here's the snippet of `--help` menu
```
Project for demoing the Affectiva Detector class (processing video files).:
  -h [ --help ]         Display this help message.
  -d [ --data ] arg     Path to the data folder. Alternatively, specify the 
                        path via the environment variable 
                        AFFECTIVA_VISION_DATA_DIR=/path/to/data
  -i [ --input ] arg    Video file to processs
  -o [ --output ] arg   Output video path.
  --sfps arg (=0)       Input sampling frame rate. Default is 0, which means 
                        the app will respect the video's FPS and read all 
                        frames
  --numFaces arg (=5)   Number of faces to be tracked.
  --loop                Loop over the video being processed.
  --face_id             Draw face id on screen. Note: Drawing to screen should 
                        be enabled.
  -q [ --quiet ]        Disable logging to console
  --occupant            Enable occupant detection
```

eg: `$ ./frame-detector-video-demo -i input_video.mp4 `

Above command will create .csv file in the same directory where the input video is present, in this case it will create input_video_faces.csv. To pull this file from the target device to host machine, do the following steps.
 
Log out from the target device.  
 `$ exit` 
 
Copy/pull csv file to current directory from target device.  
 `$ adb pull /data/local/tmp/affectiva-cpp-testapp/bin/input_video_faces.csv` 
 
If also using `-o output_video.avi` option, then `output_video.avi` file can also be pulled from target device same way  
 `$ adb pull /data/local/tmp/affectiva-cpp-testapp/bin/output_video.avi`
 

Note: 
- We currently only support avi file format for output video
- Once you logout from the target device all the environment variables are deleted. 
- This example code has been tested on the Qualcomm SA-8155P development board
