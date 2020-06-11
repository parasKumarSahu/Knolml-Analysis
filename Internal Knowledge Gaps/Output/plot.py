file_names = ["featured_top_1000", "GA", "B", "C", "Start", "Stub"]
output_1 = []
output_2 = []
output_3 = []
output_4 = []
output_5 = []

for file_name in file_names:
	f = open(file_name+".txt", encoding='utf8')
	print(file_name)
	print("==================")
	count = 1
	segments = 0
	external_gaps = 0
	edits = 0
	no_bytes = 0

	avg_know_gap = 0
	avg_flesh = 0
	avg_cli = 0
	avg_ari = 0
	avg_edit = 0

	for line in f.read().split("\n"):
		if count % 13 == 3:
			segments = int(line.split(": ")[1])
		if count % 13 == 4:
			external_gaps = int(line.split(": ")[1])
			if segments != 1:
				avg_know_gap += external_gaps/segments
			else:
				avg_know_gap += 1
		if count % 13 == 6:
			avg_flesh += float(line.split(": ")[1])
		if count % 13 == 8:
			avg_cli += float(line.split(": ")[1])
		if count % 13 == 10:
			avg_ari += float(line.split(": ")[1])
		if count % 13 == 11 and line.split(": ")[1] != 'HTML_error':
			edits = int(line.split(": ")[1])
		if count % 13 == 12 and line.split(": ")[1] != 'HTML_error':
			no_bytes = int(line.split(": ")[1])
			avg_edit += no_bytes/edits

		count += 1

	count /= 13
	count -= 1

	print("Average Knowledge Gap: ", avg_know_gap/count)
	print("Average Flesh: ", avg_flesh/count)
	print("Average CLI: ", avg_cli/count)
	print("Average ARI: ", avg_ari/count)
	print("\n")
	f.close()
	output_1.append(avg_know_gap/count)
	output_2.append(avg_flesh/count)
	output_3.append(avg_cli/count)
	output_4.append(avg_ari/count)
	output_5.append(avg_edit/count)

file_names[0] = "FA"

import matplotlib.pyplot as plt

plt.bar(file_names, output_1)
plt.xlabel("Article Category")
plt.ylabel("Average Knowledge Gap Parameter")
plt.show()
plt.close()
plt.bar(file_names, output_2)
plt.xlabel("Article Category")
plt.ylabel("Average Coffecient of Variation of Flesch–Kincaid Grade level")
plt.show()
plt.close()
plt.bar(file_names, output_3)
plt.xlabel("Article Category")
plt.ylabel("Average Coffecient of Variation of Coleman Liau readability index")
plt.show()
plt.close()
plt.bar(file_names, output_4)
plt.xlabel("Article Category")
plt.ylabel("Average Coffecient of Variation of Automated Readability Index")
plt.show()
plt.close()
plt.bar(file_names, output_5)
plt.xlabel("Article Category")
plt.ylabel("Average Edit Coffecient")
plt.show()
plt.close()
