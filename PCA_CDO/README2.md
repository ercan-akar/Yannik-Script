
# Backend
The backend is a _Flask_ REST API. It servers process data already formatted for plotting with [recharts](https://recharts.org) plotting library.
Additionally the API provides endpoints to add new process data to the current views.
By default the backend starts a seperate thread that queries data from a configured Aspen IP21 database and then posts the new live datapoints to the API.

A plant is modelled as a sequence of process steps that each are composed of multiple equipments.
An equipment is monitored by multiple sensors. These sensor values are stored in the IP21 DB under a _tag_.

## Setup
Install `python` and `pip`. It is recommended to install `python` through a bundled distribution such as `anaconda` or `miniconda`.
Install all packages listed in `requirements.txt` with

    pip install -r requirements.txt

Now you are ready to configure the server and start it with

    python server.py config_file

## Example
An example configuration together with bakers yeast example data can be found in `example/inosim`.

To start the server with example configuration run:

    python server.py example/inosim/config.json

Do not forget to adapt the path delimiters according to your OS.
