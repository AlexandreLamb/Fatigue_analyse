def make_landmarks_header:
	csv_header = []
	for i in range(1,69):
		csv_header.append("landmarks_"+str(i)+"_x")
		csv_header.append("landmarks_"+str(i)+"_y")
	return csv_header
