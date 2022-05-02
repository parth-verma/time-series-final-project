import datetime
import json

import numpy as np
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
import pandas as pd
from tensorflow import keras

# Create your views here.
df = pd.read_csv('data/txns.csv', index_col=['INSERTION_DATE'], parse_dates=['INSERTION_DATE'],
                 date_parser=pd.to_datetime)
df['TOTAL_AMOUNT'] = df['TOTAL_AMOUNT'].fillna(0).astype(float)
df['PG_TDR_SC'] = df['PG_TDR_SC'].fillna(0).astype(float)
df = df.dropna(axis=0, subset=['PAY_ID'])
df['PAY_ID'] = df['PAY_ID'].astype(int).astype(str)
pay_id = list(df['PAY_ID'].unique())
succesful_txns = df[(df['STATUS'] == 'Captured') | (df['STATUS'] == 'Settled')]

payment_type_mapping = {
    'DC': "Debit Card",
    'CC': "Credit Card",
    'NB': "Net Banking",
    'WL': "Wallet",
    'UP': "UPI",
}
import tensorflow as tf

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    0.001,
    decay_steps=400,
    decay_rate=0.98,
    staircase=True)


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)  # I use ._decayed_lr method instead of .lr

    return lr


optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
lr_metric = get_lr_metric(optimizer)


@login_required
def dashboard(request):
    active_merchant = request.GET.get('pay_id', 'All')
    start_time = pd.to_datetime(int(request.GET.get('start_date', 0)) * 10 ** 9).to_pydatetime().strftime('%Y-%m-%d')
    end_time = pd.to_datetime(
        int(request.GET.get('end_date', datetime.datetime.now().timestamp())) * 10 ** 9).to_pydatetime().strftime(
        '%Y-%m-%d')
    if not request.user.is_superuser:
        active_merchant = request.user.pay_id.pay_id

    if active_merchant == 'All':
        total_transactions = df[start_time:end_time].resample('1D').count()['TOTAL_AMOUNT']
        amount_transactions = succesful_txns[start_time:end_time].resample('1D').sum()['TOTAL_AMOUNT']
        profit = succesful_txns[start_time:end_time].resample('1D').sum()['PG_TDR_SC']
        payment_type_breakdown = succesful_txns[start_time:end_time].groupby('PAYMENT_TYPE').count()['TOTAL_AMOUNT']
        txn_count_pay_id_breakdown = succesful_txns[start_time:end_time].groupby('PAY_ID').resample('1D')[
            'TOTAL_AMOUNT'].count()
    else:
        total_transactions = \
            df[start_time:end_time][df[start_time:end_time]['PAY_ID'] == active_merchant].resample('1D').count()[
                'TOTAL_AMOUNT']
        amount_transactions = \
            succesful_txns[start_time:end_time][
                succesful_txns[start_time:end_time]['PAY_ID'] == active_merchant].resample(
                '1D').sum()[
                'TOTAL_AMOUNT']
        profit = \
            succesful_txns[start_time:end_time][
                succesful_txns[start_time:end_time]['PAY_ID'] == active_merchant].resample(
                '1D').sum()[
                'PG_TDR_SC']
        if not request.user.is_superuser:
            print('afsdfdsa')
            profit = amount_transactions - profit
        payment_type_breakdown = \
            succesful_txns[start_time:end_time][
                succesful_txns[start_time:end_time]['PAY_ID'] == active_merchant].groupby('PAYMENT_TYPE').count()[
                'TOTAL_AMOUNT']
        txn_count_pay_id_breakdown = \
            succesful_txns[start_time:end_time][
                succesful_txns[start_time:end_time]['PAY_ID'] == active_merchant].groupby('PAY_ID').resample('1D')[
                'TOTAL_AMOUNT'].count()

    payment_type_breakdown = payment_type_breakdown.to_dict()
    transaction_count = total_transactions.sum()
    transaction_amount = amount_transactions.sum()
    profit_amount = profit.sum()
    profit_transaction = {
        'x': list(map(lambda x: x.isoformat(), profit.index)),
        'y': list(profit),
        'type': 'scatter'
    }
    total_transactions = {
        'x': list(map(lambda x: x.isoformat(), total_transactions.index)),
        'y': list(total_transactions),
        'type': 'scatter'
    }
    amount_transactions = {
        'x': list(map(lambda x: x.isoformat(), amount_transactions.index)),
        'y': list(amount_transactions),
        'type': 'scatter'
    }
    payment_type_breakdown = {
        'type': 'pie',
        'values': list(payment_type_breakdown.values()),
        'labels': list(map(lambda x: payment_type_mapping[x], payment_type_breakdown.keys()))
    }
    txn_count_breakdowns = []
    for i in txn_count_pay_id_breakdown.index.get_level_values(0):
        q = txn_count_pay_id_breakdown.loc[i]
        txn_count_breakdowns.append({
            'mode': 'lines',
            'name': str(i),
            'x': list(map(lambda x: x.isoformat(), q.index)),
            'y': list(q)
        })

    return render(request, 'dashboard.html',
                  {'total_transactions': total_transactions, 'pay_ids': pay_id, 'active_merchant': active_merchant,
                   'amount_transactions': amount_transactions, 'transaction_count': format(transaction_count, ','),
                   'transaction_amount': format(round(transaction_amount, 2), ','),
                   'profit_transaction': profit_transaction, 'profit_amount': format(round(profit_amount, 2), ','),
                   'payment_type_breakdown': payment_type_breakdown, 'txn_count_breakdowns': txn_count_breakdowns})


@login_required
def predict(request):
    active_merchant = request.GET.get('pay_id', 'All')
    if not request.user.is_superuser:
        active_merchant = request.user.pay_id.pay_id
    prediction_period = 2
    if active_merchant == 'All':
        total_transactions = df.resample('1D').count()['TOTAL_AMOUNT']
        try:
            model = keras.models.load_model('model/num_txn/all')
            steps = model.layers[0].input_shape[1]
            preds = total_transactions[-steps:].tolist()
            for i in range(prediction_period):
                preds.append(model.predict(np.array(preds[-steps:]).reshape((steps, 1)))[0][0]*84)
            preds = pd.Series(preds[steps - 1:],
                              index=pd.date_range(start=total_transactions.last_valid_index(),
                                                  periods=prediction_period + 1,
                                                  freq='D'))
        except:
            preds = total_transactions[-1]
        amount_transactions = succesful_txns.resample('1D').sum()['TOTAL_AMOUNT']
        try:
            model = keras.models.load_model('model/num_txn/all')
            steps = model.layers[0].input_shape[1]
            amount_preds = amount_transactions[-steps:].tolist()
            for i in range(prediction_period):
                amount_preds.append(model.predict(np.array(amount_preds[-steps:]).reshape((steps, 1)))[0][0]*84)
            amount_preds = pd.Series(amount_preds[steps - 1:],
                                     index=pd.date_range(start=amount_transactions.last_valid_index(),
                                                         periods=prediction_period + 1,
                                                         freq='D'))
        except:
            amount_preds = total_transactions[-1]
    else:
        total_transactions = df[df['PAY_ID'] == active_merchant].resample('1D').count()['TOTAL_AMOUNT']
        try:
            model = keras.models.load_model(f'model/num_txn/{active_merchant}', custom_objects={'lr_metric': lr_metric})
            steps = model.layers[0].input_shape[1]
            preds = total_transactions[-steps:].tolist()
            for i in range(prediction_period):
                preds.append(model.predict(np.array(preds[-steps:]).reshape((steps, 1)))[0][0])
            preds = pd.Series(preds[steps - 1:],
                              index=pd.date_range(start=total_transactions.last_valid_index(),
                                                  periods=prediction_period + 1,
                                                  freq='D'))
        except Exception as e:
            print(e)
            preds = total_transactions[-1:]
        amount_transactions = succesful_txns[succesful_txns['PAY_ID'] == active_merchant].resample('1D').sum()[
            'TOTAL_AMOUNT']

        try:
            model = keras.models.load_model(f'model/num_txn/{active_merchant}', custom_objects={'lr_metric': lr_metric})
            steps = model.layers[0].input_shape[1]
            amount_preds = amount_transactions[-steps:].tolist()
            for i in range(prediction_period):
                amount_preds.append(model.predict(np.array(amount_preds[-steps:]).reshape((steps, 1)))[0][0])
            amount_preds = pd.Series(amount_preds[steps - 1:],
                                     index=pd.date_range(start=amount_transactions.last_valid_index(),
                                                         periods=prediction_period + 1,
                                                         freq='D'))
        except:
            amount_preds = amount_transactions[-1:]

    total_transactions = {
        'x': list(map(lambda x: x.isoformat(), total_transactions.index)),
        'y': list(total_transactions),
        'name': 'History',
        'type': 'scatter'
    }
    total_transactions_predictions = {
        'x': list(map(lambda x: x.isoformat(), preds.index)),
        'y': list(preds),
        'name': 'Predictions',
        'type': 'scatter'
    }
    amount_transactions = {
        'x': list(map(lambda x: x.isoformat(), amount_transactions.index)),
        'y': list(amount_transactions),
        'name': 'History',
        'type': 'scatter'
    }
    amount_transactions_predictions = {
        'x': list(map(lambda x: x.isoformat(), amount_preds.index)),
        'y': list(amount_preds),
        'name': 'Predictions',
        'type': 'scatter'
    }

    return render(request, 'predictions.html',
                  {'total_transactions': total_transactions, 'pay_ids': pay_id, 'active_merchant': active_merchant,
                   'amount_transactions': amount_transactions,
                   'amount_transactions_predictions': amount_transactions_predictions,
                   'total_transactions_predictions': total_transactions_predictions})
