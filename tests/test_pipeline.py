import unittest
import pandas as pd

from src.data_cleaning import clean_data
from src.analysis import build_master_dataset, dependency_analysis, compare_clustering_models
from src.prediction import build_monthly_salary_series, train_and_predict


def _build_raw_tables():
    employees = pd.DataFrame(
        {
            'emp_no': ['1', '2', '3', '4', '5', '6', '7', '8'],
            'birth_date': ['1970-01-01', '1978-02-02', '1980-03-03', '1985-04-04', '1988-05-05', '1975-06-06', '1982-07-07', '1984-08-08'],
            'first_name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            'last_name': ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'],
            'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
            'hire_date': ['1995-01-01', '2000-01-01', '2003-01-01', '2006-01-01', '2010-01-01', '1998-06-01', '2004-07-01', '2005-09-01'],
        }
    )
    departments = pd.DataFrame({'dept_no': ['d001', 'd002'], 'dept_name': ['Sales', 'Engineering']})
    dept_emp = pd.DataFrame(
        {
            'emp_no': ['1', '2', '3', '4', '5', '6', '7', '8'],
            'dept_no': ['d001', 'd001', 'd001', 'd001', 'd002', 'd002', 'd002', 'd002'],
            'from_date': ['1995-01-01', '2000-01-01', '2003-01-01', '2006-01-01', '2010-01-01', '1998-06-01', '2004-07-01', '2005-09-01'],
            'to_date': ['9999-01-01', '2010-01-01', '9999-01-01', '2012-01-01', '9999-01-01', '2011-01-01', '9999-01-01', '9999-01-01'],
        }
    )
    dept_manager = pd.DataFrame({'emp_no': ['1', '5'], 'dept_no': ['d001', 'd002'], 'from_date': ['2000-01-01', '2011-01-01'], 'to_date': ['9999-01-01', '9999-01-01']})
    titles = pd.DataFrame(
        {
            'emp_no': ['1', '2', '3', '4', '5', '6', '7', '8'],
            'title': ['Manager', 'Engineer', 'Engineer', 'Senior Engineer', 'Manager', 'Engineer', 'Senior Engineer', 'Engineer'],
            'from_date': ['2000-01-01', '2000-01-01', '2003-01-01', '2006-01-01', '2010-01-01', '1998-06-01', '2004-07-01', '2005-09-01'],
            'to_date': ['9999-01-01', '9999-01-01', '9999-01-01', '9999-01-01', '9999-01-01', '9999-01-01', '9999-01-01', '9999-01-01'],
        }
    )

    salary_rows = []
    for i, emp in enumerate(range(1, 9), start=1):
        for month in pd.date_range('2001-01-01', periods=36, freq='MS'):
            salary_rows.append(
                {
                    'emp_no': str(emp),
                    'salary': str(35000 + i * 1000 + (month.year - 2001) * 900 + (month.month * 10)),
                    'from_date': month.strftime('%Y-%m-%d'),
                    'to_date': '9999-01-01',
                }
            )
    salaries = pd.DataFrame(salary_rows)
    return {
        'employees': employees,
        'departments': departments,
        'dept_emp': dept_emp,
        'dept_manager': dept_manager,
        'titles': titles,
        'salaries': salaries,
    }


class PipelineTests(unittest.TestCase):
    def test_clean_analysis_and_forecast_pipeline(self):
        cleaned = clean_data(_build_raw_tables())
        master = build_master_dataset(cleaned)
        self.assertGreater(len(master), 0)
        self.assertIn('attrition_flag', master.columns)

        dep = dependency_analysis(master)
        self.assertIn('correlation_matrix', dep)
        self.assertIn('spearman_salary_vs_tenure', dep)

        clustering = compare_clustering_models(master, n_clusters=2)
        self.assertIn(clustering['preferred_model'], ['kmeans', 'agglomerative'])
        self.assertEqual(len(clustering['labels']), len(master[['salary', 'tenure_years', 'age']].dropna()))

        monthly = build_monthly_salary_series(cleaned['salaries'])
        _, validation, metrics = train_and_predict(monthly, test_periods=12)
        self.assertEqual(len(validation), 12)
        self.assertGreaterEqual(metrics['rmse'], 0.0)


if __name__ == '__main__':
    unittest.main()
