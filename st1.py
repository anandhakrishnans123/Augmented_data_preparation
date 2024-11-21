import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Streamlit UI for synthetic data generation
st.title("Synthetic Data Generator")

# Upload the Excel file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load all sheets from the Excel file
    excel_data = pd.ExcelFile(uploaded_file)
    sheets = excel_data.sheet_names

    # Multi-select box for selecting sheets
    selected_sheets = st.multiselect("Select sheets to process", sheets)

    # Dictionary to store the number of rows for each selected sheet
    rows_to_generate = {}
    columns_for_sampling = {}  # Store multiple columns for sampling
    sampling_values = {}  # Store sampling values for columns
    data_type_choices = {}  # Store data type choices for columns
    numeric_ranges = {}  # Store numeric range for columns

    if selected_sheets:
        st.markdown("### Specify the number of synthetic rows for each sheet:")
        for sheet in selected_sheets:
            # Display small title for each sheet
            st.subheader(f"Settings for Sheet: {sheet}")

            rows_to_generate[sheet] = st.number_input(
                f"Number of synthetic rows for sheet '{sheet}'",
                min_value=1,
                step=1,
                value=10,
                key=f"rows_{sheet}",
            )

            # Allow user to select multiple columns for sampling
            columns = pd.read_excel(uploaded_file, sheet_name=sheet).columns
            selected_columns = st.multiselect(
                f"Select columns to sample data for sheet '{sheet}'",
                columns,
                key=f"columns_{sheet}"
            )
            columns_for_sampling[sheet] = selected_columns

            # Allow user to input a value for each selected column for sampling
            column_values = {}
            column_data_types = {}
            column_numeric_ranges = {}
            for col in selected_columns:
                value = st.text_input(
                    f"Enter the sampling value for column '{col}' in sheet '{sheet}'",
                    key=f"value_{sheet}_{col}"
                )
                column_values[col] = value
                
                # Allow user to select the data type for each column
                data_type = st.selectbox(
                    f"Select the data type for column '{col}' in sheet '{sheet}'",
                    options=["Numeric", "Categorical", "Datetime"],
                    key=f"data_type_{sheet}_{col}"
                )
                column_data_types[col] = data_type
                
                # For numeric data type, allow user to specify a range
                if data_type == "Numeric":
                    min_value = st.number_input(
                        f"Enter the minimum value for column '{col}' in sheet '{sheet}'",
                        value=float(value) if value else 0.0,
                        key=f"min_{sheet}_{col}"
                    )
                    max_value = st.number_input(
                        f"Enter the maximum value for column '{col}' in sheet '{sheet}'",
                        value=float(value) if value else 100.0,
                        key=f"max_{sheet}_{col}"
                    )
                    column_numeric_ranges[col] = (min_value, max_value)

            sampling_values[sheet] = column_values
            data_type_choices[sheet] = column_data_types
            numeric_ranges[sheet] = column_numeric_ranges

            # Add separation between each sheet's settings
            st.markdown("---")  # Adds a horizontal line

    if st.button("Generate and Save Augmented Data"):
        # Dictionary to hold the augmented data for all sheets
        all_augmented_data = {}

        for sheet_name in selected_sheets:
            # Display small title for sheet data processing
            st.subheader(f"Processing Data for Sheet: {sheet_name}")

            # Load data from the selected sheet
            original_data = pd.read_excel(uploaded_file, sheet_name=sheet_name)

            # Exclude the header row (column names)
            data_without_header = original_data[1:]  # Skip the first row

            # Generate synthetic data
            synthetic_data = {}
            num_synthetic_rows = rows_to_generate[sheet_name]

            for column in data_without_header.columns:
                if column in columns_for_sampling[sheet_name]:
                    # Use the provided value for the selected column
                    data_type = data_type_choices[sheet_name].get(column, "Numeric")
                    value = sampling_values[sheet_name].get(column, None)

                    if data_type == "Numeric":
                        # If a numeric range is provided, generate values within the range
                        min_value, max_value = numeric_ranges[sheet_name].get(column, (0, 100))
                        synthetic_data[column] = np.random.uniform(min_value, max_value, num_synthetic_rows).tolist()

                    elif data_type == "Categorical":
                        # If the user has provided categorical values (comma-separated)
                        if value is not None:
                            unique_values = value.split(',')
                            synthetic_data[column] = [np.random.choice(unique_values) for _ in range(num_synthetic_rows)]
                    elif data_type == "Datetime":
                        # If the user has provided a datetime value (single value or a range)
                        if value is not None:
                            synthetic_data[column] = [pd.to_datetime(value)] * num_synthetic_rows

                elif pd.api.types.is_numeric_dtype(data_without_header[column]):  # Numeric columns
                    value = sampling_values[sheet_name].get(column, None)
                    if value is not None:
                        min_value, max_value = numeric_ranges[sheet_name].get(column, (0, 100))
                        synthetic_data[column] = np.random.uniform(min_value, max_value, num_synthetic_rows).tolist()
                elif (
                    isinstance(data_without_header[column].dtype, pd.CategoricalDtype)
                    or data_without_header[column].dtype == "object"
                ):  # Categorical or string columns
                    value = sampling_values[sheet_name].get(column, None)
                    if value is not None:
                        unique_values = value.split(',')
                        synthetic_data[column] = [np.random.choice(unique_values) for _ in range(num_synthetic_rows)]
                elif pd.api.types.is_datetime64_any_dtype(data_without_header[column]):  # Datetime columns
                    value = sampling_values[sheet_name].get(column, None)
                    if value is not None:
                        synthetic_data[column] = [pd.to_datetime(value)] * num_synthetic_rows
                else:
                    synthetic_data[column] = [None] * num_synthetic_rows  # Unsupported columns

            # Convert synthetic data to DataFrame
            synthetic_data_df = pd.DataFrame(synthetic_data)

            # Combine original and synthetic data
            augmented_data = pd.concat([original_data, synthetic_data_df], ignore_index=True)
            all_augmented_data[sheet_name] = augmented_data

            # Add separation after processing each sheet
            st.markdown(f"### Augmented Data for Sheet: {sheet_name}")
            st.write(augmented_data.head())  # Display the first few rows of augmented data

            # Add a line break after each sheet's preview
            st.markdown("---")

        # Save all sheets with selected sheets overwritten by augmented data
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Save all sheets, overwriting the selected ones
            for sheet in sheets:
                if sheet in all_augmented_data:
                    all_augmented_data[sheet].to_excel(writer, index=False, sheet_name=sheet)
                else:
                    # Save unmodified sheets
                    pd.read_excel(uploaded_file, sheet_name=sheet).to_excel(writer, index=False, sheet_name=sheet)

        output.seek(0)
        st.download_button(
            label="Download File with Augmented Sheets",
            data=output,
            file_name="augmented_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.success("Augmented data has been saved and is ready for download!")
