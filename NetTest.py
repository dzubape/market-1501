#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from UniformGenerator import UniformGenerator

def test_loss_acc(model, dataset_manager, test_samples_count):
    '''Tests model on train set sub-samples'''
    
    uniform_test_generator = UniformGenerator(dataset_manager, batch_size=1, epoch_batch_count=test_samples_count)
    
    y_pred = model.predict_generator(
        generator=uniform_test_generator,
        steps=test_samples_count,
        use_multiprocessing=True,
        workers=3,
        verbose=True
    ).reshape((-1,))
    
    y_true = uniform_test_generator.get_label_list(start=0, count=test_samples_count)
    
#     y_true_pred = np.stack((y_true, y_pred), axis=-1)
#     print(y_true_pred)
    
    margin = 1.
    #margin_sq = np.square((margin - y_pred > 0) * (margin - y_pred))
    margin_sq = np.square(np.asarray([(x if x > 0 else 0) for x in (margin - y_pred)]))
    #margin_sq = np.square(np.max(np.stack((np.zeros_like(y_pred), 1. - y_pred), axis=-1), axis=-1))
    pred_sq = np.square(y_pred)
    y_loss = (1 - y_true) * pred_sq + y_true * margin_sq
    
    loss_mean = np.mean(y_loss)    
    acc_mean = np.mean(y_true == (y_pred > 0.5))
        
    print("accuracy:", acc_mean)
    print("loss: ", loss_mean)
    
    
def test_base_model(model, data_generator, test_count=-1):
    
    if test_count <= 0 or test_count > data_generator.get_group_count():
        test_count = data_generator.get_group_count()
    
    ## отображаем фото сэмплов в N-мерное пространство
    predicted = model.predict_generator(
        generator=data_generator,
        steps=test_count * 2,
        use_multiprocessing=True,
        workers=3,
        verbose=True
    )
    print("Размерность пространства отображения: {}".format(predicted[0].shape))
    
    shape = list(predicted.shape)
    shape[0] = -1    
    shape.insert(1, 2)
    
    predicted = predicted.reshape(shape)
    print("Классов в сравнении {}".format(len(predicted)))
        
    ## Euclidean distance for N-dim
    def eucl_dist(a, b):
        c = a - b
        c_sq = np.square(c)
        c_sum = np.sum(c_sq)
        return np.sqrt(c_sum)
    
    ## euclidean dist ##
    y_pred = []
    y_true = []
    test_person_list = []
    for i in range(len(predicted)):
        base_vec = predicted[i][0] ## вектор для базового изображения класса
        sim_vec = predicted[i][1] ## вектор для изображения того же класса
        sim_dist = eucl_dist(base_vec, sim_vec) ## евкл. расстояние между векторами изображений одного класса
        sim_list = [sim_dist] ## группа расстояний между соседними векторами
        
        y_pred.append(sim_dist)
        y_true.append(0)
        
        ## расстояние до иных объектов
        dis_list = np.ndarray(len(predicted) - 1, dtype=np.float32)
        shift = 0
        for j, [dis_vec, _] in enumerate(predicted):
            if i == j:
                shift = 1
                continue
            dis_dist = eucl_dist(base_vec, dis_vec)
            dis_list[j - shift] = dis_dist
            
            y_pred.append(dis_dist)
            y_true.append(1)
        dis_list = np.sort(dis_list)
        
        
        test_person_list.append({
            "sim": sim_list,
            "dis": dis_list
        })
    
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_acc = np.mean(y_true == (y_pred > 0.5))
    print("accuracy: {}".format(y_acc))
    
    ## Отрисовка распределения ошибок sim/dis ##
    fig = plt.figure(figsize=(4, 2), dpi=112)
    fig_title = "1024-128-sgd-200, Error spread ({} gr.)".format(test_count)
    y_end = np.stack((y_pred, y_true, y_true == (y_pred > 0.5)), axis=-1)
    y_sim_err = {i: trio[0] for i, trio in enumerate(y_end) if not trio[2] and not trio[1]}
    y_dis_err = {i: trio[0] for i, trio in enumerate(y_end) if not trio[2] and trio[1]}
    
    err_ax = plt.subplot(1, 1, 1)
    err_ax.set_title(fig_title)
    err_ax.plot(y_pred, "y,")
    err_ax.plot(list(y_sim_err.keys()), list(y_sim_err.values()), "g+")    
    err_ax.plot(list(y_dis_err.keys()), list(y_dis_err.values()), "r,")
    fig.savefig("{}.png".format(fig_title))
    ####
    
    ## формируем ранги ##
    rank_list = np.full((len(test_person_list),), len(test_person_list), dtype=np.int32)
    for i, group in enumerate(test_person_list):
        sim_dist = group["sim"][0]
        for j, dis_dist in enumerate(group["dis"]):
            if sim_dist <= dis_dist:
                rank_list[i] = j
                break
    
    #print("rank_list:", rank_list)
    
    ## выявляем ранг для каждого класса ##
    cmc_dict = dict()
    for rank in rank_list:
        if rank not in cmc_dict:
            cmc_dict[rank] = 1
        else:
            cmc_dict[rank] += 1
            
    worst_rank = max(cmc_dict.keys())
    rank_rate_list = np.full((worst_rank + 1,), 0, dtype=np.int32)
    for rank, rate in cmc_dict.items():
        rank_rate_list[rank] = rate
    
    cmc_list = np.zeros_like(rank_rate_list, dtype=np.float32)
    rate_acc = 0
    for rank, rate in enumerate(rank_rate_list):
        if rate > 0:
            rate_acc += rate
        cmc_list[rank] = rate_acc
    cmc_list /= test_count
    
    ## отрисовка графика CMC ##
    fig = plt.figure(figsize=(10, 3), dpi=112)
    fig_title = "1024-128-sgd-200, CMC ({} gr.)".format(test_count)
    cmc_ax = plt.subplot(1, 1, 1)
    cmc_ax.set_xlabel("rank")
    cmc_ax.set_ylabel("rate")
    cmc_ax.set_title(fig_title)
    cmc_ax.plot(cmc_list)    
    fig.savefig("{}.png".format(fig_title))
    ####
    
    acc_rank = 0
    for rank, rate in enumerate(cmc_list):   
        if rate > y_acc:
            break
        acc_rank = rank
    plt.plot(acc_rank, y_acc, "ro")