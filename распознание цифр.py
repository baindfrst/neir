import numpy
import scipy.special
# pip install scipy
def init_net():
    input_nodes = 784
    print('Введите число скрытых нейронов: ')
    hidden_nodes = int(input())
    out_nodes = 10
    print('Введите скорость обучения(0.5): ')
    lern_node = float(input())
    return input_nodes, hidden_nodes, out_nodes, lern_node
def creat_net(input_nodes, hidden_nodes, out_nodes,):
    # сознание массивов. -0.5 вычитаем что бы получить диапазон -0.5 +0.5 для весов
    input_hidden_w = (numpy.random.rand(hidden_nodes, input_nodes) - 0.5)
    hidden_out_w = (numpy.random.rand(out_nodes, hidden_nodes) - 0.5)
    return input_hidden_w, hidden_out_w

def fun_active(x):
    return scipy.special.expit(x)

def query(input_hidden_w, hidden_out_w, inputs_list):
    # преобразовать список входных значений
    # в двухмерный массив
    inputs_sig = numpy.array(inputs_list, ndmin=2).T

    hidden_inputs = numpy.dot(input_hidden_w, inputs_sig)  # умножение матриц
    hidden_out = fun_active(hidden_inputs)  # вычисляем выходной сигнал скрытого слоя
    # умножение матриц выходи в веса для выходного слоя
    final_inputs = numpy.dot(hidden_out_w, hidden_out)
    final_out = fun_active(final_inputs)

    return final_out

def treyn(targget_list,input_list, input_hidden_w, hidden_out_w, lern_node):
    #Прогоняем данные через сеть
    targgets = numpy.array(targget_list, ndmin=2).T
    inputs_sig = numpy.array(input_list, ndmin=2).T
    hidden_inputs = numpy.dot(input_hidden_w, inputs_sig)
    hidden_out = fun_active(hidden_inputs)
    final_inputs = numpy.dot(hidden_out_w, hidden_out)
    final_out = fun_active(final_inputs)
    #Рассчитываем ошибку выходного слоя
    out_errors = targgets - final_out
    #Рассчитываем ошибку скрытого слоя
    hidden_errors = numpy.dot(hidden_out_w.T, out_errors)
    # Обновление весов связей
    hidden_out_w += lern_node * numpy.dot((out_errors * final_out * (1 - final_out)), numpy.transpose(hidden_out))
    input_hidden_w += lern_node * numpy.dot((hidden_errors * hidden_out * (1 - hidden_out)),numpy.transpose(inputs_sig))
    return hidden_out_w, input_hidden_w

def test_set(hidden_out_w, input_hidden_w):
    data_file = open('mnist_train.csv', 'r')
    trening_list = data_file.readlines()
    data_file.close()

    for record in trening_list:
        all_values = record.split(',')
        # масштабировать и сместить входные значения
        # numpy.asfarray(a,dtype=float64'>>)  Возвращает массив,преобразованный в тип float.
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # Создаем массив из 10 элементов и в нужном элемент (считанный из первого знака базы данных) записываем проверочное значение.
        targets = numpy.zeros(10) + 0.01
        targets[int(all_values[0])] = 0.99
        hidden_out_w, input_hidden_w = treyn(targets, inputs, input_hidden_w, hidden_out_w, lern_node)

    data_file = open('mnist_test.csv', 'r')
    test_list = data_file.readlines()
    data_file.close()
    test = []
    for record in test_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        out_session = query(input_hidden_w, hidden_out_w, inputs)
        if int(all_values[0]) == numpy.argmax(out_session):
            test.append(1)
        else:
            test.append(0)
    print("выполнено тетов : ", len(test))
    test = numpy.asarray(test)
    print('Эфективность сети % =', (test.sum() / test.size) * 100)
    return hidden_out_w, input_hidden_w
input_nodes, hidden_nodes, out_nodes, lern_node = init_net()
input_hidden_w, hidden_out_w = creat_net(input_nodes, hidden_nodes, out_nodes)
for i in range (5):
    print('Test #', i+1)
    hidden_out_w, input_hidden_w = test_set(hidden_out_w, input_hidden_w)
def work(hidden_out_w, input_hidden_w, record):
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    out_session = query(input_hidden_w, hidden_out_w, inputs)
    return out_session
print("обучение завершино")
inwork = input('введите название файла, с его расширение (пример: input.txt (файл должен находиться в папке с программой)). Для выхода напишить "stop" : ')
if inwork != "stop":
    work_input = open(inwork, 'r')
    record = work_input.readline()
    print("заданное число: ", numpy.argmax(work(hidden_out_w, input_hidden_w, record)))