from data_preprocessing import load_data
from model import create_model

def main():
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Create model
    model = create_model()

    # Train model
    model.fit(x_train, y_train, epochs=5)

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')

    # Save the model
    model.save('handwritten_digit_model.h5')

if __name__ == '__main__':
    main()
