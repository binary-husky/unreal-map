from flask import Flask, url_for, jsonify, request, send_from_directory, redirect
import json, glob
import numpy as np
import re, os
from multiprocessing import Process, Pipe



