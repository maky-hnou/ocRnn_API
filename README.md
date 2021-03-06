# ocRnn_API

![Python version][python-version]
[![GitHub issues][issues-image]][issues-url]
[![GitHub forks][fork-image]][fork-url]
[![GitHub Stars][stars-image]][stars-url]
[![License][license-image]][license-url]

## About this repo:  

A REST API for Optical Character Recognition.  

The API has been developed using *Django Framework*, while the OCR was created based on *RNN and LSTM*. The OCR repository can be found [here](https://github.com/maky-hnou/ocRnn).  

## Content of the repo:  
The project has been organized as follows:  
- `manage.py`: the python script used to run django commands (`runserver/makemigrations/maigrate` and the commands added by the the user).  
- `requirements.txt`: a text file containing the needed packages to run the project.  
- `OcRnn/`: the folder containing the model,views, urls, serializer, migration files, the custom commands and the core of the app.  
- `OcRnn/core`: the folder containing the model and the code used for the OCR.  
- `ocRnn/management/`: the folder that contains the custom commands.  
- `ocRnn/migrations/`: the folder containing the migration files.  
- `ocRnn_API/`: the folder that contains the api settings and the urls.  
- `media/`: the folder where the uploaded images will be saved.  
- `templates/`: the folder that contains the HTML, CSS and Javscript files.  

## Run the app:  
*N.B:* use Python 3.8+  

**1. Clone the repo:**  
on your terminal, run `git clone https://github.com/maky-hnou/ocRnn_API.git`  
Then get into the project folder: `cd ocRnn_API/`  
We need to install some dependencies:  
`sudo apt install python3-pip libpq-dev python3-dev`

**2. Install requirements:**  
Before running the app, we need to install some packages.  
- *Optional* Create a virtual environment:  To do things in a clean way, let's create a virtual environment to keep things isolated.  
Install the virtual environment wrapper: `pip3 install virtualenvwrapper`  
Add the following lines to `~/.bashrc`:  
```
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=~/.local/bin/virtualenv
source ~/.local/bin/virtualenvwrapper.sh
```
Run `source ~/.bashrc`  
Run `mkvirtualenv ocrnn`  
Activate the virtual environment: `workon ocrnn` (To deactivate the virtual environment, run `deactivate`)  
- Install requirements: To install the packages needed to run the application, run `pip3 install -r requirements.txt`  

**3. Install postgreSQL:**  
We need to install postgresql: `sudo apt install postgresql postgresql-contrib`  
go to the lines 81/82/83 in `ocRnn_API/settings.py` and choose the database name, user and password.  
After that we create a database, user and password based on the ones you chose.  
```
sudo -u postgres psql
create database db_name;
create user db_user with encrypted password 'db_password';
grant all privileges on database db_name to db_user;
\q
```  

Make sure to change `db_name, db_user and db_password` by the ones you chose in settings.py.  
You may need to apply the changes you did by running `python3 manage.py migrate`.
Then create a super user using: `python manage.py createsuperuser` and follow the steps.  
Once we finish, we can run our app.  

**4. Run the app:**  
To run the application, you need to run the following command: `python3 manage.py runserver`.  
Then on your browser, open http://127.0.0.1:8000/.  
You'll be asked to enter a login and a password, enter the ones you chose while creating a super user.  

**5. Use the app:**  
After logging in, click on image models to open the interface.  
You can upload an image by clicking on *ADD IMAGE MODEL* on the top right.  
Once you add an image, click on *save*.  
To run the classification, check the box near the ID of the image you uploaded, then in *Actions*, choose *Run ocr* and then press Go.  
After that, the page will reload and you'll get a bunch of information including the extracted text.  
To delete an image, check the box near its ID and choose *Delete selected image models* then press Go.  
It is also possible to run the ocr from the terminal by opening another terminal while the app is running and run the following command (make sure to activate the virtual environment):  
`python manage.py run_ocr`  

**6. Other options:**  
You can filter the uploaded images by 'processed/not processed' or by upload date.  

**7. Demo:**  

<a href="https://github.com/maky-hnou/ocRnn_API/blob/main/demo/demo.gif"><img src="https://github.com/maky-hnou/ocRnn_API/blob/main/demo/demo.gif" title="demo-gif"/></a>

[python-version]:https://img.shields.io/badge/python-3.8+-brightgreen.svg
[issues-image]:https://img.shields.io/github/issues/maky-hnou/ocRnn_API.svg
[issues-url]:https://github.com/maky-hnou/ocRnn_API/issues
[fork-image]:https://img.shields.io/github/forks/maky-hnou/ocRnn_API.svg
[fork-url]:https://github.com/maky-hnou/ocRnn_API/network/members
[stars-image]:https://img.shields.io/github/stars/maky-hnou/ocRnn_API.svg
[stars-url]:https://github.com/maky-hnou/ocRnn_API/stargazers
[license-image]:https://img.shields.io/github/license/maky-hnou/ocRnn_API.svg
[license-url]:https://github.com/maky-hnou/ocRnn_API/blob/main/LICENSE
