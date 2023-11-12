from model import Model

import sys


if __name__ == "__main__":
	model = Model()
	params = {
		'video_path': sys.argv[1],
		'language': sys.argv[2]
	}

	model.run(params)
