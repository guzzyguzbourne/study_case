# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 02:03:06 2022

@author: gzner
"""
from flask import *
import pandas as pd
import train_model
import xlrd

# creating the flask object
app = Flask(__name__)

REQUIRED_COLUMNS = ['account_amount_added_12_24m', 'account_days_in_dc_12_24m',
       'account_days_in_rem_12_24m', 'account_days_in_term_12_24m', 'age',
       'avg_payment_span_0_12m', 'avg_payment_span_0_3m', 'has_paid',
       'max_paid_inv_0_12m', 'max_paid_inv_0_24m',
       'num_active_div_by_paid_inv_0_12m', 'num_active_inv',
       'num_arch_dc_0_12m', 'num_arch_ok_12_24m', 'num_arch_ok_0_12m', 'num_arch_rem_0_12m',
       'num_arch_written_off_0_12m', 'num_unpaid_bills',
       'status_last_archived_0_24m', 'status_2nd_last_archived_0_24m',
       'status_3rd_last_archived_0_24m', 'status_max_archived_0_6_months',
       'status_max_archived_0_12_months', 'status_max_archived_0_24_months',
       'sum_capital_paid_account_0_12m', 'sum_capital_paid_account_12_24m', 'sum_paid_inv_0_12m', 'time_hours',
       'merchant_group', 'merchant_category']

# for each required columns we need to
@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('upload_file.html')


ALLOWED_EXTENSIONS = {'xlsx'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])
def home_page():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            if allowed_file(file.filename):
                df = pd.read_excel(request.files.get('file'),  engine = 'openpyxl')
                for col in REQUIRED_COLUMNS:
                    if col not in df.columns:
                        return "One more required columns are missing. List of required columns are = {} ".format(REQUIRED_COLUMNS)
                    
                # 2. Data prepartion
                df_final = train_model.prepare_data(df)
                    
                # 3. Prediction
                prediction = train_model.predict(df_final)
                return prediction.to_html(header="true", table_id="table")

        else:
            return render_template('error_message.html')
    else:
        return render_template('error_message.html')




if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)