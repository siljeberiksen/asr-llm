
import json
import matplotlib.pyplot as plt
import statistics

#with open("result/wer_nb_samtale_5_tiny_2.json", 'r') as file:
with open("result/wer_nb_samtale_llm_5_tiny_10_prompt_3.json", 'r') as file:
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
differnece_beams_cer =[]
count_files = 0
for data in wer_data:
    if count_files > 7:
        break
    if(data["sentence_order"]==0):
        count_files +=1
    wer_best_beam.append(min(data["wer"]))
    cer_best_beam.append(min(data["cer"]))
    wer_worst_beam.append(max(data["wer"]))
    cer_worst_beam.append(max(data["cer"]))
    difference_beams.append(max(data["wer"])-min(data["wer"]))
    differnece_beams_cer.append(max(data["cer"])-min(data["cer"]))

    wer_acutal.append(data["wer_result"])
    cer_actual.append(data["cer_result"])

    wer_sum += data["wer_result"]
    cer_sum += data["cer_result"]
    length +=1
wer_avg= wer_sum / length
cer_avg = wer_sum / length

x =  list(range(1, length))
print("Average", sum(difference_beams)/length)
#print(difference_beams)

print(length)

print("Count", difference_beams.count(0))

print("Average cer", sum(differnece_beams_cer)/length)
print("Count", differnece_beams_cer.count(0))
median_value = statistics.median(wer_acutal)
print("median_wer", median_value)

median_value= statistics.median(cer_actual)
print("median_cer", median_value)

print(wer_sum/length)
print(cer_sum/length)

plt.plot(x, wer_best_beam, label='Best beam option', color='blue', linestyle='--')
plt.plot(x, wer_worst_beam, label='Worst beam option', color='red', linestyle='--')
plt.plot(x, wer_acutal, label='Actual output', color='green', linestyle='--')

plt.title('WER Best, Worst, and Actual Output')
plt.xlabel('Sample Index')
plt.ylabel('WER')

plt.legend()

plt.show()

