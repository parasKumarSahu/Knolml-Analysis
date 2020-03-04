from readability import Readability
import sys

#Main Function
if len(sys.argv) < 2:
    print("Input Format: python3 script_name input_file_name")
    exit()
    
book_path = sys.argv[1]
text = open(book_path).read()

r = Readability(text)

fk = r.flesch_kincaid()
print("Score:", fk.score)
print("Grade Level:", fk.grade_level)

cl = r.coleman_liau()
print("Score:", cl.score)
print("Grade Level:", cl.grade_level)

ari = r.ari()
print("Score:", ari.score)
print("Ages:", ari.ages)
