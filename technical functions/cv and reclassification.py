import numpy as np
import pandas as pd

def errors_pack(conf_mat_with_total):
    omission_err_list = []
    commission_err_list = []
    prod_accuracy_list = []
    user_accuracy_list = []

    conf_mat = conf_mat_with_total[:-1, :-1]
    for i in range(0, len(conf_mat)):
        err_ind = [j != i for j in range(0, len(conf_mat))]
        om_err = sum(conf_mat[err_ind, i]) / float(conf_mat_with_total[-1, i])
        omission_err_list.append(om_err)
        com_err = sum(conf_mat[i, err_ind]) / float(conf_mat_with_total[i, -1])
        commission_err_list.append(com_err)
        prod_acc = conf_mat[i, i] / float(conf_mat_with_total[-1, i])
        prod_accuracy_list.append(prod_acc)
        user_acc = conf_mat[i, i] / float(conf_mat_with_total[i, -1])
        user_accuracy_list.append(user_acc)
    return omission_err_list, commission_err_list, prod_accuracy_list, user_accuracy_list
  
  def confusion_matrix(y_real, y_pred, keys_list):
    # формирование матрицы
    conf_mat = metrics.confusion_matrix(y_real, y_pred).T
    unique_values = np.unique(np.concatenate((y_pred, y_real)))
    if len(unique_values) < len(keys_list):
        for i in range(0, len(keys_list)):
            if i not in unique_values:
                conf_mat = np.insert(np.insert(conf_mat, i, np.zeros((1, len(conf_mat))), axis=0), i,
                                     np.zeros((1, len(conf_mat) + 1)), axis=1)

    # добавление сумм
    conf_mat_with_total = np.concatenate((conf_mat, np.zeros((1, len(conf_mat)))))
    conf_mat_with_total = np.concatenate((conf_mat_with_total, np.zeros((len(conf_mat_with_total), 1))), axis=1)
    conf_mat_with_total[:-1, -1] = np.sum(conf_mat_with_total[:-1, :-1], axis=1)
    conf_mat_with_total[-1, :] = np.sum(conf_mat_with_total[:-1, :], axis=0)

    return conf_mat_with_total
 
def model_test(estimator, X_train, y_train, keys_list=None, name=None, directory_path=None, cross_validation=True,
               reclassification=True, cv=10, stratified=False):
    # функция, выделяющая дагональ таблицы (правильные результаты) и общих сумм (Total)
    def diagonal_excretion_and_total(table):
        style_table = table.copy()
        style_table.loc[:, :] = ''
        table_type = list(style_table.columns)[0][0]
        keys = list(style_table.loc['Predicted', (table_type, 'Real')])
        # выделение диагонали
        for key in keys:
            style_table.loc[('Predicted', key), (table_type, 'Real', key)] = 'color: green'
        # выделение Total
        style_table.loc[('Predicted', 'Total')] = 'background-color : yellow; color : black'
        style_table.loc[:, (table_type, 'Real', 'Total')] = 'background-color : yellow; color : black'
        style_table.loc[
            ('Predicted', 'Total'), (table_type, 'Real', 'Total')] = 'background-color : orange; color : black;'
        return style_table
    # функция, выделяющая значения больше нуля
    def up_zero_excretion(val):
        color = 'red' if val > 0 else 'black'
        return 'color: %s' % color

    conf_mat_data_frame_list = []
    acc_err_data_frame_list = []
    total_coef_data_frame_list = []

    if keys_list is None:
        keys_list = sorted(list(set(y_train)))

    if cross_validation:
        if stratified:
            kf = model_selection.KFold(cv, shuffle=True)
        else:
            kf = model_selection.StratifiedKFold(cv, shuffle=True)
        y_real = []
        y_pred = []
        for train_index, test_index in kf.split(X_train, y_train):
            X_cv_train = X_train[train_index]
            y_cv_train = y_train[train_index]
            estimator.fit(X_cv_train, y_cv_train)
            y_real += list(y_cv_train)
            y_pred += list(estimator.predict(X_cv_train))

        # формирование матрицы
        conf_mat_with_total = confusion_matrix(y_real, y_pred, keys_list)

        # вычисление общей точности с учетом возможности отказа от классификации
        overall_accuracy = metrics.accuracy_score(y_real, y_pred)
        # вычисление статистики Каппа
        cohen_kappa = metrics.cohen_kappa_score(y_real, y_pred)

        # определение ошибкок пропуска цели и ошибкок ложной тревоги, producer’s accuracy и user’s accuracy
        omission_err_list, commission_err_list, prod_accuracy_list, user_accuracy_list = errors_pack(
            conf_mat_with_total)

        # запись данных в таблицы
        # таблица для confusion matrix
        # заголовки
        type_name = str(cv) + '-fold cross-validation'
        labels = np.concatenate((keys_list, ['Total']))
        indexes = pd.MultiIndex.from_product([['Predicted'], labels])
        header = pd.MultiIndex.from_product([[type_name], ['Real'], labels])

        # запись матрицы в DataFrame
        conf_mat_data_frame = pd.DataFrame(np.array(conf_mat_with_total), index=indexes, columns=header)
        # форматирование: выделение диагонали таблицы (правильной классификации) зеленым и ошибок больше нуля
        #   (неправильной классификации) красным
        conf_mat_data_frame = conf_mat_data_frame.style.applymap(up_zero_excretion).apply(
            diagonal_excretion_and_total,
            axis=None)
        conf_mat_data_frame_list.append(conf_mat_data_frame)

        # таблица с точностью и ошибками для каждого класса
        header = pd.MultiIndex.from_product([[type_name],
                                             ['Omission error', 'Commission error', 'Producer’s accuracy',
                                              'User’s accuracy']])
        acc_err_data_frame = pd.DataFrame(
            np.array([omission_err_list, commission_err_list, prod_accuracy_list, user_accuracy_list]).T,
            columns=header,
            index=keys_list)
        acc_err_data_frame_cv = acc_err_data_frame
        acc_err_data_frame_list.append(acc_err_data_frame)

        # таблица с общими коэффициентами (также записывается ошибка кросс-валидации)
        header = pd.MultiIndex.from_product([[type_name], ['Value']])
        total_coef_data_frame = pd.DataFrame(np.array([overall_accuracy, cohen_kappa]).T,
                                             columns=header,
                                             index=['Accuracy', 'Kappa Coefficient'])
        total_coef_data_frame_list.append(total_coef_data_frame)

    if reclassification:
        y_pred = estimator.predict(X_train)

        # формирование матрицы
        conf_mat_with_total = confusion_matrix(y_train, y_pred, keys_list)

        # вычисление общей точности с учетом возможности отказа от классификации
        overall_accuracy = metrics.accuracy_score(y_train, y_pred)
        # вычисление статистики Каппа
        cohen_kappa = metrics.cohen_kappa_score(y_train, y_pred)

        # определение ошибкок пропуска цели и ошибкок ложной тревоги, producer’s accuracy и user’s accuracy
        omission_err_list, commission_err_list, prod_accuracy_list, user_accuracy_list = errors_pack(
            conf_mat_with_total)

        # запись данных в таблицы
        # таблица для confusion matrix
        # заголовки
        labels = np.concatenate((keys_list, ['Total']))
        indexes = pd.MultiIndex.from_product([['Predicted'], labels])
        header = pd.MultiIndex.from_product([['Reclassification'], ['Real'], labels])

        # запись матрицы в DataFrame
        conf_mat_data_frame = pd.DataFrame(np.array(conf_mat_with_total), index=indexes, columns=header)
        # форматирование: выделение диагонали таблицы (правильной классификации) зеленым и ошибок больше нуля
        #   (неправильной классификации) красным
        conf_mat_data_frame = conf_mat_data_frame.style.applymap(up_zero_excretion).apply(
            diagonal_excretion_and_total,
            axis=None)
        conf_mat_data_frame_list.append(conf_mat_data_frame)

        # таблица с точностью и ошибками для каждого класса
        header = pd.MultiIndex.from_product(
            [['Reclassification'], ['Omission error', 'Commission error', 'Producer’s accuracy', 'User’s accuracy']])
        acc_err_data_frame = pd.DataFrame(
            np.array([omission_err_list, commission_err_list, prod_accuracy_list, user_accuracy_list]).T,
            columns=header,
            index=keys_list)
        acc_err_data_frame_rcls = acc_err_data_frame
        acc_err_data_frame_list.append(acc_err_data_frame)

        # таблица с общими коэффициентами (также записывается ошибка кросс-валидации)
        header = pd.MultiIndex.from_product([['Reclassification'], ['Value']])
        total_coef_data_frame = pd.DataFrame(np.array([overall_accuracy, cohen_kappa]).T,
                                             columns=header,
                                             index=['Accuracy', 'Kappa Coefficient'])
        total_coef_data_frame_list.append(total_coef_data_frame)
    return conf_mat_data_frame, conf_mat_data_frame_list, acc_err_data_frame_list, total_coef_data_frame_list, acc_err_data_frame_cv,  acc_err_data_frame_rcls
