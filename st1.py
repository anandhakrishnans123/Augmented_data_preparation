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
    column_for_sampling = {}  # Store columns for sampling
    sampling_value = {}  # Store sampling values for columns
    
    if selected_sheets:
        
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

            # Allow user to select a specific column for sampling
            columns = pd.read_excel(uploaded_file, sheet_name=sheet).columns
            selected_column = st.selectbox(
                f"Select a column to sample data for sheet '{sheet}'",
                columns,
                key=f"column_{sheet}"
            )
            column_for_sampling[sheet] = selected_column

            # Allow user to input custom values (beyond existing) for sampling
            if selected_column == "Department":  # Specific case for "Department" column
                # Get existing unique values
                existing_values = pd.read_excel(uploaded_file, sheet_name=sheet)[selected_column].dropna().unique().tolist()

                # Allow user to input new values beyond the existing ones
                new_values = st.text_area(
                    f"Enter new comma-separated values to add for column '{selected_column}' in sheet '{sheet}'",
                    key=f"value_{sheet}"
                )

                # Split input into a list of new values
                if new_values:
                    new_values_list = new_values.split(",")
                    new_values_list = [value.strip() for value in new_values_list]  # Clean extra spaces
                else:
                    new_values_list = []

                # Combine existing and new values
                all_values = list(set(existing_values + new_values_list))  # Remove duplicates
                sampling_value[sheet] = all_values

            else:
                # Option 2: Use multiselect for other columns
                input_values = st.multiselect(
                    f"Select sampling values for column '{selected_column}' in sheet '{sheet}'",
                    options=pd.read_excel(uploaded_file, sheet_name=sheet)[selected_column].dropna().unique(),
                    key=f"multi_value_{sheet}"
                )
                sampling_value[sheet] = input_values

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
                if column == column_for_sampling[sheet_name]:
                    # Use the sampling values for the selected column
                    if len(sampling_value[sheet_name]) > 0:
                        synthetic_data[column] = np.random.choice(sampling_value[sheet_name], size=num_synthetic_rows)
                    else:
                        synthetic_data[column] = [None] * num_synthetic_rows
                elif pd.api.types.is_numeric_dtype(data_without_header[column]):  # Numeric columns
                    synthetic_data[column] = np.random.normal(
                        loc=data_without_header[column].mean(),
                        scale=data_without_header[column].std(),
                        size=num_synthetic_rows,
                    )
                elif (
                    isinstance(data_without_header[column].dtype, pd.CategoricalDtype)
                    or data_without_header[column].dtype == "object"
                ):  # Categorical or string columns
                    unique_values = data_without_header[column].dropna().unique()
                    if len(unique_values) > 0:
                        probabilities = (
                            data_without_header[column]
                            .dropna()
                            .value_counts(normalize=True)
                            .reindex(unique_values, fill_value=0)
                            .values
                        )
                        synthetic_data[column] = np.random.choice(
                            unique_values, size=num_synthetic_rows, p=probabilities
                        )
                    else:
                        synthetic_data[column] = [None] * num_synthetic_rows
                elif pd.api.types.is_datetime64_any_dtype(data_without_header[column]):  # Datetime columns
                    if not data_without_header[column].isna().all():
                        synthetic_data[column] = pd.to_datetime(
                            np.random.choice(
                                pd.date_range(
                                    start=data_without_header[column].min(),
                                    end=data_without_header[column].max(),
                                ),
                                size=num_synthetic_rows,
                            )
                        )
                    else:
                        synthetic_data[column] = [None] * num_synthetic_rows
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
