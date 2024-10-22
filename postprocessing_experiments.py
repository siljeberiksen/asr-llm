
import json
import matplotlib.pyplot as plt

with open("result/wer_nb_samtale_5_tiny_2.json", 'r') as file:
    wer_data = json.load(file)

wer_best_beam = []
cer_best_beam = []
wer_worst_beam = []
cer_worst_beam = []
wer_sum = 0
cer_sum = 0
wer_acutal =[]
cer_actual = []
first_started = False
length = 1
difference_beams = []
for data in wer_data:
    if(data["sentence_order"] == 0 and first_started):
        break
    if(data["sentence_order"] == 0):
        first_started = True
    wer_best_beam.append(min(data["wer"]))
    cer_best_beam.append(min(data["cer"]))
    wer_worst_beam.append(max(data["wer"]))
    cer_worst_beam.append(max(data["cer"]))
    difference_beams.append(max(data["wer"])-min(data["wer"]))

    wer_acutal.append(data["wer_result"])
    cer_actual.append(data["cer_result"])

    wer_sum += data["wer_result"]
    cer_sum += data["cer_result"]
    length +=1
wer_avg= wer_sum / length
cer_avg = wer_sum / length

x =  list(range(1, length))
print(sum(difference_beams)/length)
print(difference_beams)
print(length)

print(difference_beams.count(0))

plt.plot(x, wer_best_beam, label='Best beam option', color='blue', linestyle='--')
plt.plot(x, wer_worst_beam, label='Worst beam option', color='red', linestyle='--')
plt.plot(x, wer_acutal, label='Actual output', color='green', linestyle='--')

plt.title('WER Best, Worst, and Actual Output')
plt.xlabel('Sample Index')
plt.ylabel('WER')

plt.legend()

plt.show()

