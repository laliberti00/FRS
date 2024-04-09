import src.data_processing.filter_divide_dataset
from src.data_processing.filter_divide_dataset import main as filter_divide_dataset
from src.fuzzy_user_signature.user_signature import main as user_signature
from src.fuzzy_user_signature.kindredness_calculation import main as kindredness_calculation
from src.predictions.predictions_maker import main as predictions_maker
from src.evaluation.predictions_evaluation import main as predictions_evaluation
def main():
    print(1)
    filter_divide_dataset()
    print(2)
    user_signature()
    print(3)
    kindredness_calculation(10)
    print(4)
    predictions_maker()
    print(5)
    predictions_evaluation()

if __name__ == "__main__":
    main()