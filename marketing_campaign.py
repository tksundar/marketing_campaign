from Analysis import *
from Tests import *
from  Visualisations import *

def analyse_data(mkt_data):
    print('*******************************************************************************')
    print('Starting basis data analysis for preliminary insights')
    print('*******************************************************************************')
    analysis = Analysis(mkt_data)
    analysis.grouped_bar_chart()
    analysis.plot_box_and_histogram()
    analysis.plot_heatmaps()

def do_hypothesis_tests(data):
    print('*******************************************************************************')
    print('Starting hypothesis testing')
    print('*******************************************************************************')
    tests = Tests(data)
    tests.test_older_individuals_rely_on_store_shopping()
    tests.test_kids_influence_online_shopping()
    tests.test_cannibalization_by_off_store_sales_channels()
    tests.test_us_outperforms_purchases()


def do_visualisations(data):
    print('*******************************************************************************')
    print('Starting Visualisations for correlations')
    print('*******************************************************************************')
    visualise = Visualisations(data)
    visualise.top_performer()
    visualise.correlation_customer_age_and_campaign()
    visualise.country_with_highest_campaign_response()
    visualise.correlation_num_kids_and_total_spend()
    visualise.correlation_between_education_and_complaints()
    visualise.correlation_of_spending_patterns()


menu = ['Analyse Data','Do hypothesis Tests','Do Correlations','Perform All Tasks']

def perform_all_tasks(mkt_data):
    analyse_data(mkt_data)
    do_hypothesis_tests(mkt_data)
    do_visualisations(mkt_data)

if __name__ == '__main__':
    print('*******************************************************************************')
    print('Marketing Campaign Project')
    print('*******************************************************************************')
    mkt_data = clean_data(pd.read_csv('marketing_data.csv'))
    for i,v in enumerate(menu):
        print('%d %s' %(i+1, v))

    choice = input('\nEnter your choice : ')

    if choice == '1':
        analyse_data(mkt_data)
    elif choice == '2':
        do_hypothesis_tests(mkt_data)
    elif choice == '3':
        do_visualisations(mkt_data)
    elif choice == '4':
        perform_all_tasks(mkt_data)
    else:
        raise ValueError('Illegal input')


