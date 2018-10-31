import time
import subprocess
import os
import platform

SERVER_READY_STRING = "ready"

def build_command(image, keypoints, depth, affine, result):
	return " ".join((image, keypoints, depth, affine, result))

def server_executable_name():
	NAME = "FaceWarperServer"
	if platform.system() == 'Windows':
		return NAME + '.exe'
	else:
		NAME

class Server:
	def __init__(self, exec_path=None, working_dir=None):
		self._exec_path = exec_path
		if exec_path is None:
			self._exec_path = server_executable_name()
		self._working_dir = working_dir
		self._proc = None

	def send_command(self, command):
		self._proc.stdin.write(command + "\n")
		self._proc.stdin.flush()
		while True:
			line = self._proc.stdout.readline()
			if SERVER_READY_STRING in line:
				break
			time.sleep(0.001)

	def _connect(self):
		self._proc = subprocess.Popen([self._exec_path], cwd=self._working_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
		# Wait for server to be ready
		while True:
			line = self._proc.stdout.readline()
			if SERVER_READY_STRING in line:
				break
			time.sleep(0.001)

	def _terminate(self):
		if self._proc is not None:
			self._proc.terminate()
			self._proc = None

	def __enter__(self):
		self._connect()
		return self

	def __exit__(self, *args):
		self._terminate()
