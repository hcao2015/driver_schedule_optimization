# **Uber Driver Schedule Optimization README**
Optimization for Uber driver schedule in a week based on driver preferences on:

1. Days of the week availability
2. Maximum number of hours working per week
3. Location availability
4. Time availability

### **Environment Configuration**

This assumes you have already cloned this repo to your local machine.

1. `pyenv virtualenv driver_schedule_env`
2. Activate your virtual environment by running `pyenv activate driver_schedule_env`
3. Install package dependencies `pipenv install`
4. Exit your virtual environment by running `pyenv deactivate`

### **Run example**

From the project directory, run the script in solver.py as:

```bash
python solver.py --max-hours-weekly 12 --time-available 16 17 18 19 --day-available 0 1 3 4 --location-available 0 9 18
```