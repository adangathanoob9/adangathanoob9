from src.data_preprocessing import load_and_preprocess
from src.model_training      import train_model
from src.evaluation          import evaluate_model
from src.visualization       import (plot_satisfaction_distribution,
                                     plot_feature_importance,
                                     plot_satisfaction_by_travel_type,
                                     plot_delay_vs_satisfaction,
                                     plot_service_heatmap)
from src.insights             import run_all_insights

if __name__ == '__main__':
    print("\n🚀 Starting Airline Satisfaction Analysis...\n")

    # Step 1 – Load & preprocess
    X_train, X_test, y_train, y_test, df = load_and_preprocess('data/train.csv')

    # Step 2 – Train
    model = train_model(X_train, y_train)

    # Step 3 – Evaluate
    y_pred, acc = evaluate_model(model, X_test, y_test)

    # Step 4 – Visualize
    plot_satisfaction_distribution(df)
    plot_feature_importance(model, X_train)
    plot_satisfaction_by_travel_type(df)
    plot_delay_vs_satisfaction(df)
    plot_service_heatmap(df)

    # Step 5 – Insights + PDF Report (Step 9)
    run_all_insights(df, model, X_train, acc)

    print("\n🎉 Project Complete! Check outputs/ folder for all results.")