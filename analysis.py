import streamlit as st
import pandas as pd
import plotly.express as px
import io
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly

# --- Page Configuration ---
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Helper Function to Load Main Sales Data ---
@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses the uploaded sales XLSX file."""

    # The 26 columns from your sales file, in order.
    SALES_COLUMN_NAMES = [
        "No", "Branch", "Customer/Debtor", "Sales Invoice No", "Sale Date", 
        "Sale Time", "Salesperson", "Product Barcode", "Product Code", 
        "Product Name", "Unit", "Quantity Sold", "Unit Price Before Discount", 
        "Discount per Unit", "Unit Price After Discount", "Cost per Unit", 
        "Profit per Unit", "Total Before Discount", "Total Discount", 
        "Total After Discount", "Total Cost", "Total Profit", "Profit %", 
        "Markup %", "Added By", "Status"
    ]
    
    # Map from your names to the names the script expects
    SALES_COLUMNS_RENAME_MAP = {
        "Sale Date": "Sales Date",
        "Sale Time": "Sales Time",
        "Sales Invoice No": "Invoice No",
        "Total Before Discount": "Total Price Before Discount",
        "Total After Discount": "Total Price After Discount"
        # Any names that already match (e.g., "Product Name") don't need to be here
    }

    try:
        # 1. Read the Excel file (first sheet, first row is header)
        df = pd.read_excel(uploaded_file, sheet_name=0, header=0)
        
        # 2. Critical Check: Ensure we have AT LEAST the expected columns
        if len(df.columns) < len(SALES_COLUMN_NAMES):
            st.error(f"Error: The uploaded sales file has only {len(df.columns)} columns, but the dashboard expects at least {len(SALES_COLUMN_NAMES)}.")
            st.error("Please ensure you've uploaded the correct, unmodified sales export.")
            return None
            
        # 3. Slice the df to only include the columns we want (removes extra blank columns)
        df = df.iloc[:, :len(SALES_COLUMN_NAMES)]
        
        # 4. Rename columns by their position
        df.columns = SALES_COLUMN_NAMES

        # 5. Rename to match script's internal names
        df.rename(columns=SALES_COLUMNS_RENAME_MAP, inplace=True)

        # --- Critical Data Preprocessing (This part is identical to before) ---
        
        # 1. Combine Date and Time and create datetime object
        if 'Sales Date' not in df.columns or 'Sales Time' not in df.columns:
            st.error("Error: 'Sales Date' or 'Sales Time' column not found after renaming.")
            return None
            
        # Convert time to string just in case it's read as a time object
        df['Sales Time'] = df['Sales Time'].astype(str)
        
        df['datetime'] = pd.to_datetime(
            df['Sales Date'].astype(str).str.split(' ').str[0] + ' ' + df['Sales Time'], 
            errors='coerce',
        )
        
        # 2. Drop rows where date/time conversion failed
        df.dropna(subset=['datetime'], inplace=True)
        
        if df.empty:
            st.error("Error: All date parsing failed. Please check your source file's date format.")
            return None
        
        # 3. Extract day of week and hour
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['hour_of_day'] = df['datetime'].dt.hour
        
        # 4. Convert numeric columns
        numeric_cols = [
            'Quantity Sold', 'Unit Price Before Discount', 'Discount per Unit',
            'Unit Price After Discount', 'Cost per Unit', 'Profit per Unit',
            'Total Price Before Discount', 'Total Discount', 'Total Price After Discount',
            'Total Cost', 'Total Profit'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 5. Handle potential missing invoice numbers for transaction counting
        if 'Invoice No' not in df.columns:
            st.warning("Warning: 'Invoice No' not found. Using row index for transaction counting.")
            df['Invoice No'] = df.index
        
        # 6. Create 'Product Barcode' if it's missing (shouldn't be, but good failsafe)
        if 'Product Barcode' not in df.columns:
            df['Product Barcode'] = df['Product Code'].fillna('NO_BARCODE')
            
        df.dropna(subset=['Total Profit', 'Quantity Sold', 'Invoice No'], inplace=True)

        # 7. Select only the columns needed for the dashboard to ensure clean concatenation
        required_cols = [
            'datetime', 'Product Name', 'Quantity Sold', 'Total Price After Discount',
            'Total Cost', 'Total Profit', 'Product Barcode', 'Invoice No',
            'day_of_week', 'hour_of_day'
        ]
        
        # Ensure all required columns exist, add placeholders if not
        for col in required_cols:
            if col not in df.columns:
                # Add a placeholder, e.g., 0 for numeric, empty string for text
                if col in ['Quantity Sold', 'Total Price After Discount', 'Total Cost', 'Total Profit']:
                    df[col] = 0
                else:
                    df[col] = 'N/A'
        
        return df[required_cols]

    except Exception as e:
        st.error(f"Error loading sales data: {e}")
        st.error("Please ensure 'openpyxl' is installed (`pip install openpyxl`) and the file is a valid .xlsx file.")
        return None

# --- [NEW] Helper Function to Load UTC Data ---
@st.cache_data
def load_utc_data(uploaded_file):
    """Loads and preprocesses the uploaded UTC (CSV) file."""
    try:
        df = pd.read_csv(uploaded_file)
        
        # 1. Parse Timestamp (Format: 25/10/2025 17:51:53)
        if 'Timestamp' not in df.columns:
            st.error("UTC Data Error: 'Timestamp' column not found.")
            return None
        df['datetime'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
        
        # 2. Rename columns to match main data
        UTC_COLUMNS_RENAME_MAP = {
            "Product": "Product Name",
            "Amount Sold": "Quantity Sold",
            "Total Revenue": "Total Price After Discount",
            "Total Cost": "Total Cost",
            "Total Profit": "Total Profit"
        }
        df.rename(columns=UTC_COLUMNS_RENAME_MAP, inplace=True)
        
        # 3. Handle missing essential columns
        if 'Product Name' not in df.columns:
            st.error("UTC Data Error: 'Product' column not found.")
            return None
        
        # Fill 'Amount Gifted' with 0 if it exists, to add to Quantity Sold
        if 'Amount Gifted' in df.columns:
            df['Amount Gifted'] = pd.to_numeric(df['Amount Gifted'], errors='coerce').fillna(0)
            df['Quantity Sold'] = pd.to_numeric(df['Quantity Sold'], errors='coerce').fillna(0)
            df['Quantity Sold'] = df['Quantity Sold'] + df['Amount Gifted']
        
        
        # 4. Create placeholder columns needed for the dashboard
        
        # Create a unique barcode for each UTC product
        df['Product Barcode'] = "UTC_" + df['Product Name'].str.replace(' ', '_')
        
        # Create a unique invoice number for each row (since each row is a sale)
        df['Invoice No'] = [f"UTC_INV_{i}" for i in range(len(df))]
        
        # 5. Extract day of week and hour
        df.dropna(subset=['datetime'], inplace=True)
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['hour_of_day'] = df['datetime'].dt.hour
        
        # 6. Convert numeric columns
        numeric_cols = ['Quantity Sold', 'Total Price After Discount', 'Total Cost', 'Total Profit']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                st.warning(f"UTC Data Warning: Missing numeric column '{col}'. Setting to 0.")
                df[col] = 0 # Add column as 0 if it's missing
        
        df.dropna(subset=['Total Profit', 'Quantity Sold', 'Invoice No', 'datetime'], inplace=True)
        
        # 7. Select only the columns needed for concatenation
        required_cols = [
            'datetime', 'Product Name', 'Quantity Sold', 'Total Price After Discount',
            'Total Cost', 'Total Profit', 'Product Barcode', 'Invoice No',
            'day_of_week', 'hour_of_day'
        ]
        
        return df[required_cols]

    except Exception as e:
        st.error(f"Error loading UTC data: {e}")
        st.error("Please ensure it's a valid .csv file with the correct format.")
        return None

        
@st.cache_data
def load_stock_data(stock_file):
    """Loads and processes the uploaded stock XLSX file."""

    # The 32 columns from your stock file, in order.
    STOCK_COLUMN_NAMES = [
        "No.", "Branch Name", "Main Supplier", "Product Code", "Barcode", 
        "Product Name", "Stock", "Average Cost", "Selling Price 1", "Profit", 
        "Profit %", "Total Sales (1)", "Total Cost", "Total Profit", "Unit", 
        "Account Period Cost", "Latest Purchase Cost", "Total Stock (All Units)", 
        "Generic Drug Name", "Storage", "Product Type", "Product Group", "Brand", 
        "Selling Price 2", "Selling Price 3", "Selling Price 4", "Selling Price 5", 
        "Tax Exempt Product", "Decimal Places", "Status", "Name 2", "Name 3"
    ]
    
    try:
        # 1. Read the Excel file
        df = pd.read_excel(stock_file, sheet_name=0, header=0)

        # 2. Critical Check: Ensure we have AT LEAST the expected columns
        if len(df.columns) < len(STOCK_COLUMN_NAMES):
            st.error(f"Error: The uploaded stock file has only {len(df.columns)} columns, but the dashboard expects at least {len(STOCK_COLUMN_NAMES)}.")
            st.error("Please ensure you've uploaded the correct, unmodified stock export.")
            return None
            
        # 3. Slice the df to only include the columns we want (removes extra blank columns)
        df = df.iloc[:, :len(STOCK_COLUMN_NAMES)]
        
        # 4. Rename columns by their position
        df.columns = STOCK_COLUMN_NAMES
        
        # --- Final check ---
        if 'Barcode' not in df.columns or 'Stock' not in df.columns or 'Product Name' not in df.columns:
            st.error("Stock file is missing 'Barcode', 'Stock', or 'Product Name' after renaming. Please check the column order.")
            return None
        
        # --- Select and clean required columns ---
        df_stock = df[['Barcode', 'Stock', 'Product Name']].copy()
        df_stock['Stock'] = pd.to_numeric(df_stock['Stock'], errors='coerce')

        # --- FIX: Convert Barcode from float to int to string ---
        df_stock['Barcode'] = df_stock['Barcode'].astype(str).str.strip()
        df_stock['Barcode'] = pd.to_numeric(df_stock['Barcode'], errors='coerce')
        df_stock.dropna(subset=['Stock', 'Barcode'], inplace=True)
        df_stock['Barcode'] = df_stock['Barcode'].astype('Int64').astype(str)
        
        # In case a barcode is listed multiple times, sum its stock
        df_stock_agg = df_stock.groupby('Barcode').agg(
            Stock=('Stock', 'sum'),
            product_name=('Product Name', 'first') # Get the first name associated
        ).reset_index()
        
        return df_stock_agg

    except Exception as e:
        st.error(f"Error loading stock data: {e}. Please ensure it's a CSV or TSV file.")
        return None
        
@st.cache_data
def get_transaction_kpis(df):
    """Calculates key transaction KPIs."""
    unique_invoices = df['Invoice No'].nunique()
    if unique_invoices == 0:
        return 0, 0, 0
    
    total_revenue = df['Total Price After Discount'].sum()
    total_items = df['Quantity Sold'].sum()
    
    aov = total_revenue / unique_invoices
    items_per_transaction = total_items / unique_invoices
    
    return aov, items_per_transaction, unique_invoices

@st.cache_data
def get_common_co_purchases(df, selected_product, top_n=5):
    """Finds the most commonly co-purchased items."""
    # 1. Find all invoices containing the selected product
    target_invoices = df[df['Product Name'] == selected_product]['Invoice No'].unique()
    
    if len(target_invoices) == 0:
        return pd.DataFrame(columns=['Product Name', 'count']) # Return empty
    
    # 2. Get all items from those invoices
    co_purchase_df = df[df['Invoice No'].isin(target_invoices)]
    
    # 3. Exclude the selected product itself
    other_items_df = co_purchase_df[co_purchase_df['Product Name'] != selected_product]
    
    if other_items_df.empty:
        return pd.DataFrame(columns=['Product Name', 'count']) # Return empty
    
    # 4. Count the other items
    top_items = other_items_df['Product Name'].value_counts().head(top_n).reset_index()
    top_items.columns = ['Product Name', 'count']
    
    return top_items

# --- [REMOVED] Flawed calculate_future_demand function ---

# --- Main App ---
st.title("🛒 Sales & Inventory Dashboard")

# --- Sidebar for File Upload ---
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("1. Upload Sales Data (.xlsx)", type=["xlsx"])
utc_file = st.sidebar.file_uploader("3. Upload UTC Data (.csv)", type=["csv"])
stock_file = st.sidebar.file_uploader("2. Upload Stock Data (.xlsx)", type=["xlsx"])

# --- Load and Process Data ---
df_main = None
df_utc = None
all_dataframes = []

if uploaded_file is not None:
    df_main = load_data(uploaded_file)
    if df_main is not None:
        all_dataframes.append(df_main)

if utc_file is not None:
    df_utc = load_utc_data(utc_file)
    if df_utc is not None:
        all_dataframes.append(df_utc)


if not all_dataframes:
    st.info("Please upload your sales data file (and optionally UTC data) using the sidebar to get started.")
else:
    # --- Combine all loaded dataframes ---
    df = pd.concat(all_dataframes, ignore_index=True)
    
    # --- Calculate num_days based on the COMBINED dataframe ---
    num_days = df['datetime'].dt.date.nunique()
    if num_days == 0:
        num_days = 1 # Avoid division by zero

    # Add messages to the sidebar to debug the date range
    st.sidebar.info(f"Combined data range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    st.sidebar.success(f"Calculating velocity based on **{num_days}** unique sales day(s).")
    
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    
    # --- [MODIFIED] Inverse Filter by Product ---
    all_products = df['Product Name'].unique()
    excluded_products = st.sidebar.multiselect(
        "Select products to EXCLUDE from dashboard:", # <-- CHANGED
        options=all_products,
        default=[] # <-- CHANGED
    )
    
    # Filter data based on sidebar selection
    df_filtered = df[~df['Product Name'].isin(excluded_products)] # <-- CHANGED (added ~)
    
    # --- [NEW] Global Stock Configuration (Moved from Tab 5) ---
    st.sidebar.header("Stock Configuration")
    lead_time_weeks = st.sidebar.number_input(
        "Weeks of stock to keep on hand?", 
        min_value=1, max_value=12, value=3, step=1
    )
    
    if df_filtered.empty:
        st.warning("No data found for the selected filters.")
    else:
        # --- Create Tabs ---
        tab_list = [
            "📈 Overall Trend & Forecasting",  # <-- MOVED TO FIRST
            "🔥 Busiest Times Heatmap",      # <-- MOVED TO SECOND
            "🏆 Product Performance", 
            "🛒 Customer Purchase Insights", 
            "📦 Inventory Insights"
        ]
        
        # Dynamically add the new tab if stock file is present
        if stock_file is not None:
            tab_list.append("🛍️ Reorder & Stock Check")
        
        tabs = st.tabs(tab_list)

        # --- Tab 1: Overall Trend & Forecasting (NEW FIRST TAB) ---
        with tabs[0]:
            st.header("Overall Sales Trend & Forecasting")

            # --- [NEW] Top-Level KPIs ---
            st.subheader("Period-Wide KPIs")
            st.markdown("These metrics reflect the *entire filtered dataset*.")
            
            # 1. Calculate main totals
            total_revenue = df_filtered['Total Price After Discount'].sum()
            total_profit = df_filtered['Total Profit'].sum()
            total_cost = df_filtered['Total Cost'].sum()

            # 2. Calculate daily stats (for trend plot and prophet)
            # --- [MODIFIED] ---
            df_daily_stats = df_filtered.groupby(df_filtered['datetime'].dt.date) \
                                      .agg(
                                          daily_sales=('Total Price After Discount', 'sum'),
                                          daily_profit=('Total Profit', 'sum') # <-- ADDED
                                      ) \
                                      .reset_index()
            df_daily_stats = df_daily_stats.rename(columns={'datetime': 'date'})
            # --- [END MODIFIED] ---
            
            # 3. Get transaction KPIs
            aov, items_per_tx, total_tx = get_transaction_kpis(df_filtered)

            # 4. Calculate remaining KPIs
            avg_daily_sales = total_revenue / num_days if num_days > 0 else 0
            profit_margin_pct = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0
            markup_pct = (total_profit / total_cost) * 100 if total_cost > 0 else 0

            # 5. Display KPIs in columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Revenue", f"฿{total_revenue:,.2f}")
            col2.metric("Total Profit", f"฿{total_profit:,.2f}")
            col3.metric("Total Cost", f"฿{total_cost:,.2f}")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Total Transactions", f"{total_tx:,}")
            col5.metric("Average Order Value (AOV)", f"฿{aov:,.2f}")
            col6.metric("Avg. Items per Transaction", f"{items_per_tx:,.2f}")
            
            col7, col8, col9 = st.columns(3)
            col7.metric("Average Daily Sales", f"฿{avg_daily_sales:,.2f}")
            col8.metric("Profit Margin", f"{profit_margin_pct:,.2f}%")
            col9.metric("Markup", f"{markup_pct:,.2f}%")
            
            st.markdown("---") # Separator

            # --- 1. Prepare data for Trend Plot ---
            # --- [MODIFIED] ---
            st.subheader("Overall Daily Sales & Profit Trend")
            # The df_daily_stats calculation is already done above for the KPIs

            # Melt the dataframe for plotting
            df_melted = df_daily_stats.melt(
                id_vars=['date'], 
                value_vars=['daily_sales', 'daily_profit'], 
                var_name='Metric', 
                value_name='Amount (฿)'
            )
            
            # Rename for a cleaner legend
            df_melted['Metric'] = df_melted['Metric'].map({
                'daily_sales': 'Total Sales',
                'daily_profit': 'Total Profit'
            })

            # Plot overall trend
            fig_trend = px.line(
                df_melted,
                x='date',
                y='Amount (฿)',
                color='Metric', # <-- Use color to differentiate
                title="Total Daily Sales & Profit Over Time",
                labels={'date': 'Date', 'Amount (฿)': 'Amount (฿)'},
                color_discrete_map={ # <-- Assign colors as requested
                    'Total Sales': 'blue', # (Default, but good to be explicit)
                    'Total Profit': 'green' # <-- User requested green
                }
            )
            fig_trend.update_layout(xaxis_title="Date", yaxis_title="Amount (฿)")
            st.plotly_chart(fig_trend, use_container_width=True)
            # --- [END MODIFIED] ---

            # --- [NEW] Daily Transactions Bar Chart ---
            st.subheader("Overall Daily Transactions Trend")
            
            # 1. Aggregate transactions (nunique invoices) by date
            df_daily_transactions = df_filtered.groupby(df_filtered['datetime'].dt.date) \
                                              .agg(daily_transactions=('Invoice No', 'nunique')) \
                                              .reset_index()
            df_daily_transactions = df_daily_transactions.rename(columns={'datetime': 'date'})

            # 2. Plot the bar chart
            fig_bar_transactions = px.bar(
                df_daily_transactions,
                x='date',
                y='daily_transactions',
                title="Total Transactions Per Day",
                labels={'date': 'Date', 'daily_transactions': 'Total Transactions'}
            )
            fig_bar_transactions.update_layout(xaxis_title="Date", yaxis_title="Number of Transactions")
            st.plotly_chart(fig_bar_transactions, use_container_width=True)


            # --- 2. Prophet Forecasting ---
            st.subheader("Sales Forecast with Prophet")
            
            # Format for Prophet: needs 'ds' and 'y'
            # --- [MODIFIED] ---
            df_prophet_revenue = df_daily_stats[['date', 'daily_sales']].rename(
                columns={'date': 'ds', 'daily_sales': 'y'}
            )
            # --- [END MODIFIED] ---
            
            # --- [REMOVED] logic for df_prophet_units ---
            
            # Check for sufficient data
            if len(df_prophet_revenue) < 5:
                st.warning("Not enough daily data to generate a forecast. Please provide data spanning at least 5 different days.")
            else:
                # Forecasting parameters
                forecast_days = st.slider("Days to forecast into the future", 7, 365, 30, key="forecast_days_slider")
                
                # Cache the forecast function
                @st.cache_data
                def get_prophet_forecast(data, periods):
                    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
                    m.fit(data)
                    future = m.make_future_dataframe(periods=periods)
                    forecast = m.predict(future)
                    return m, forecast
                
                # --- [REMOVED] get_prophet_unit_growth_k function ---

                # Run forecast
                with st.spinner(f"Generating {forecast_days}-day forecast..."):
                    try:
                        # --- Run the REVENUE forecast for plotting ---
                        m_revenue, forecast_revenue = get_prophet_forecast(df_prophet_revenue, forecast_days)
                        
                        # --- [REMOVED] All logic for unit_growth_k ---
                        
                        st.subheader(f"{forecast_days}-Day Sales (Revenue) Forecast")
                        fig_forecast = plot_plotly(m_revenue, forecast_revenue)
                        fig_forecast.update_layout(
                            title=f"Daily Sales (Revenue) Forecast",
                            xaxis_title="Date",
                            yaxis_title="Forecasted Sales (฿)"
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        st.subheader("Revenue Forecast Components")
                        # Prophet's component plot uses matplotlib, so we use st.pyplot
                        fig_components = m_revenue.plot_components(forecast_revenue)
                        st.pyplot(fig_components)
                    
                    except Exception as e:
                        st.error(f"An error occurred during forecasting: {e}")
                        st.error("This can happen if there isn't enough varied data (e.g., all sales on one day).")


        # --- Tab 2: Busiest Times Heatmap (NEW SECOND TAB) ---
        with tabs[1]:
            st.header("Busiest Times Heatmap")
            st.markdown("This heatmap shows the number of unique transactions per hour and day of the week.")
            
            # Filter for the hours you requested (9am to 8pm -> 9 to 20)
            df_heatmap = df_filtered[
                (df_filtered['hour_of_day'] >= 9) & (df_filtered['hour_of_day'] <= 20)
            ]

            if df_heatmap.empty:
                st.warning("No sales data found between 9 AM and 8 PM.")
            else:
                # Group by day and hour, counting unique invoices
                heatmap_data = df_heatmap.groupby(['day_of_week', 'hour_of_day'])['Invoice No'].nunique().reset_index()
                
                # Pivot the data for the heatmap
                heatmap_pivot = heatmap_data.pivot(
                    index='day_of_week',
                    columns='hour_of_day',
                    values='Invoice No'
                ).fillna(0) # Fill empty slots with 0 transactions
                
                # Order the days and hours correctly
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                hour_order = list(range(9, 21)) # 9am to 8pm
                
                heatmap_pivot = heatmap_pivot.reindex(index=day_order, columns=hour_order, fill_value=0)
                
                # Create the Plotly heatmap
                fig = px.imshow(
                    heatmap_pivot,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Transactions"),
                    x=heatmap_pivot.columns,
                    y=heatmap_pivot.index,
                    text_auto=True, # Show the numbers on the squares
                    aspect="auto"
                )
                fig.update_layout(
                    title="Hourly Sales Transactions by Day",
                    xaxis_nticks=len(hour_order),
                    yaxis_title=None,
                    xaxis_title="Hour of Day (24h format)"
                )
                st.plotly_chart(fig, use_container_width=True)

        # --- Tab 3: Product Performance ---
        with tabs[2]:
            st.header("Product Performance")
            st.markdown(f"Analyzed over **{num_days}** days of sales.")
            
            # Group data by product
            product_performance = df_filtered.groupby('Product Name').agg(
                total_quantity_sold=('Quantity Sold', 'sum'),
                total_profit=('Total Profit', 'sum'),
                total_revenue=('Total Price After Discount', 'sum')
            ).reset_index()
            
            # --- Top Products ---
            st.subheader("Top Performing Products")
            
            # Select metric to sort by
            sort_by = st.selectbox("Sort top products by:", ["Total Profit", "Total Quantity Sold", "Total Revenue"], index=0)
            
            if sort_by == "Total Profit":
                sorted_products = product_performance.sort_values(by='total_profit', ascending=False)
                chart_col = 'total_profit'
            elif sort_by == "Total Quantity Sold":
                sorted_products = product_performance.sort_values(by='total_quantity_sold', ascending=False)
                chart_col = 'total_quantity_sold'
            else:
                sorted_products = product_performance.sort_values(by='total_revenue', ascending=False)
                chart_col = 'total_revenue'

            # --- KPIs ---
            col1, col2 = st.columns(2)
            if not sorted_products.empty:
                best_product = sorted_products.iloc[0]
                col1.metric(
                    f"Best Product (by {sort_by})",
                    best_product['Product Name'],
                )
                col2.metric(
                    f"Value (by {sort_by})",
                    f"฿{best_product[chart_col]:,.2f}" if chart_col != 'total_quantity_sold' else f"{best_product[chart_col]:,.0f} Units"
                )
            else:
                col1.metric(f"Best Product (by {sort_by})", "N/A")
                col2.metric(f"Value (by {sort_by})", "N/A")


            # --- Bar Chart of Top 10 ---
            st.subheader("Top 10 Products")
            top_10_products = sorted_products.head(10)
            
            if top_10_products.empty:
                st.warning("No product data to display.")
            else:
                fig_bar = px.bar(
                    top_10_products,
                    x='Product Name',
                    y=chart_col,
                    title=f"Top 10 Products by {sort_by}",
                    labels={'Product Name': 'Product', chart_col: sort_by},
                    text_auto=True
                )
                fig_bar.update_layout(xaxis_title=None)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # --- Full Data Table ---
            with st.expander("View All Product Performance Data"):
                st.dataframe(sorted_products, use_container_width=True)

        # --- Tab 4: Customer Purchase Insights ---
        with tabs[3]:
            st.header("🛒 Customer Purchase Insights")
            st.markdown("Understand how customers bundle items in a single transaction.")
            
            # --- 1. Calculate and display KPIs ---
            aov, items_per_tx, total_tx = get_transaction_kpis(df_filtered)
            st.subheader("High-Level Metrics")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Order Value (AOV)", f"฿{aov:,.2f}")
            col2.metric("Avg. Items per Transaction", f"{items_per_tx:,.2f}")
            col3.metric("Total Transactions", f"{total_tx:,}")

            st.markdown("---")
            
            # --- 2. Co-Purchase Finder ---
            st.subheader("What products are bought together?")
            st.markdown("Select a product to see the top 5 other items most frequently purchased in the same transaction.")
            
            # Get a list of top 200 products by sales for the dropdown to keep it manageable
            top_product_list = df_filtered['Product Name'].value_counts().head(200).index.tolist()
            
            if not top_product_list:
                st.warning("No products available to select.")
            else:
                selected_product = st.selectbox(
                    "Select a product:",
                    options=top_product_list
                )
                
                if selected_product:
                    # Get the co-purchase data
                    co_purchase_data = get_common_co_purchases(df_filtered, selected_product, top_n=5)
                    
                    if co_purchase_data.empty:
                        st.info(f"No other products were frequently purchased with '{selected_product}'.")
                    else:
                        # Create the bar chart
                        fig_co = px.bar(
                            co_purchase_data,
                            x='count',
                            y='Product Name',
                            orientation='h',
                            title=f"Top 5 Products Bought With '{selected_product}'",
                            labels={'count': 'Number of Transactions', 'Product Name': 'Product'}
                        )
                        fig_co.update_layout(yaxis=dict(autorange="reversed")) # Show top item at the top
                        st.plotly_chart(fig_co, use_container_width=True)

        # --- Tab 5: Inventory Insights ---
        with tabs[4]:
            st.header("Inventory Insights & Sales Velocity")
            
            # --- [REMOVED] Prophet info message ---
            
            st.markdown(f"Calculations are based on the total sales over **{num_days}** days of data provided.")
            
            # Group by Barcode and Name
            group_cols = ['Product Name', 'Product Barcode']

            inventory_stats = df_filtered.groupby(group_cols).agg(
                total_quantity_sold=('Quantity Sold', 'sum'),
            ).reset_index()
            
            # --- Rename 'Product Barcode' to 'Barcode' for merging ---
            inventory_stats.rename(columns={'Product Barcode': 'Barcode'}, inplace=True)
            
            # --- FIX: Robust Barcode Cleaning ---
            # Ensure Barcode is string for merging, even for UTC items
            inventory_stats['Barcode'] = inventory_stats['Barcode'].astype(str).str.strip()
            
            # Calculate sales velocity
            inventory_stats['avg_daily_sales'] = inventory_stats['total_quantity_sold'] / num_days
            inventory_stats['avg_weekly_sales'] = inventory_stats['avg_daily_sales'] * 7
            
            # Add stocking suggestion
            st.subheader("Stocking Suggestions")
            # [MODIFIED] lead_time_weeks is now in the sidebar.
            # --- [FIXED] Updated helper text ---
            st.markdown(f"`Suggested Stock Level` is based on **{lead_time_weeks} weeks** of safety stock (set in sidebar), calculated from your average weekly sales.")
            
            # --- [FIXED] Reverted to simple linear calculation ---
            inventory_stats['suggested_stock_level'] = inventory_stats['avg_weekly_sales'] * lead_time_weeks
            
            # --- [NEW] Round up suggested stock ---
            inventory_stats['suggested_stock_level'] = np.ceil(inventory_stats['suggested_stock_level'])
            
            # Format for display
            inventory_display = inventory_stats.copy()
            
            st.dataframe(
                inventory_display.sort_values(by='avg_weekly_sales', ascending=False),
                use_container_width=True,
                column_config={
                    "Product Name": st.column_config.TextColumn("Product Name", width="large"),
                    "Barcode": st.column_config.TextColumn("Barcode"),
                    "total_quantity_sold": st.column_config.NumberColumn("Total Sold", format="%d units"),
                    "avg_daily_sales": st.column_config.NumberColumn("Avg. Daily Sales", format="%.2f"),
                    "avg_weekly_sales": st.column_config.NumberColumn("Avg. Weekly Sales", format="%.2f"),
                    "suggested_stock_level": st.column_config.NumberColumn(f"Suggested {lead_time_weeks}-Week Stock", format="%.0f units"),
                }
            )
        
        # --- Tab 6: Reorder & Stock Check (Conditional) ---
        if stock_file is not None:
            with tabs[5]:
                st.header("Stock vs. Forecasted Demand")
                
                # Load stock data
                df_stock = load_stock_data(stock_file)
                
                if df_stock is not None:
                    
                    # --- [NEW] Filter out UTC items for this tab ONLY ---
                    # We identify UTC items by their 'Product Barcode' starting with "UTC_"
                    # We must filter df_filtered *before* grouping.
                    df_restock_sales_data = df_filtered[
                        ~df_filtered['Product Barcode'].astype(str).str.startswith('UTC_')
                    ]
                    
                    if df_restock_sales_data.empty:
                        st.warning("No non-UTC sales data found to calculate restock levels.")
                        # Stop execution for this tab if no data
                    else:
                        # --- [NEW] Recalculate inventory_stats from non-UTC data ---
                        group_cols = ['Product Name', 'Product Barcode']
                        inventory_stats_restock = df_restock_sales_data.groupby(group_cols).agg(
                            total_quantity_sold=('Quantity Sold', 'sum'),
                        ).reset_index()

                        inventory_stats_restock.rename(columns={'Product Barcode': 'Barcode'}, inplace=True)
                        inventory_stats_restock['Barcode'] = inventory_stats_restock['Barcode'].astype(str).str.strip()

                        # --- [NEW] Calculate velocity from this new dataframe ---
                        inventory_stats_restock['avg_daily_sales'] = inventory_stats_restock['total_quantity_sold'] / num_days
                        inventory_stats_restock['avg_weekly_sales'] = inventory_stats_restock['avg_daily_sales'] * 7
                        
                        # --- [FIXED] Reverted to simple linear calculation ---
                        inventory_stats_restock['suggested_stock_level'] = inventory_stats_restock['avg_weekly_sales'] * lead_time_weeks
                        
                        # --- [NEW] Round up suggested stock ---
                        inventory_stats_restock['suggested_stock_level'] = np.ceil(inventory_stats_restock['suggested_stock_level'])

                        # --- [MODIFIED] Merge using the new 'inventory_stats_restock' ---
                        df_merged = pd.merge(
                            inventory_stats_restock, # <-- CHANGED
                            df_stock[['Barcode', 'Stock']], # From stock data
                            on='Barcode',
                            how='left' 
                        )
                        
                        # Calculate the deficit
                        # Fillna(0) for items sold but not in stock file
                        df_merged['Stock'] = df_merged['Stock'].fillna(0)
                        df_merged['deficit'] = df_merged['suggested_stock_level'] - df_merged['Stock']
                        
                        # --- [NEW] Round up deficit ---
                        df_merged['deficit'] = np.ceil(df_merged['deficit'])
                        
                        st.subheader("Reorder List")
                        # --- [FIXED] Updated helper text ---
                        st.markdown(f"This list shows items where your `Current Stock` is *less* than the `Suggested Stock Level`. (Suggested level is based on **{lead_time_weeks} weeks** of safety stock, set in the sidebar).")
                        
                        # Filter for items that need reordering
                        df_reorder = df_merged[df_merged['deficit'] > 0].sort_values(by='deficit', ascending=False)
                        
                        # --- [NEW] "Already Ordered" Ticking logic ---
                        all_reorder_products = df_reorder['Product Name'].unique()
                        
                        if 'ordered_list' not in st.session_state:
                            st.session_state.ordered_list = []

                        # Filter session state to only include items still in the reorder list
                        st.session_state.ordered_list = [
                            p for p in st.session_state.ordered_list if p in all_reorder_products
                        ]

                        ordered_items = st.multiselect(
                            "Mark items as 'Already Ordered' (to hide from list):",
                            options=all_reorder_products,
                            key='ordered_list' # Use session state to remember
                        )
                        
                        # Filter the dataframe to hide the "ticked" items
                        df_reorder_to_display = df_reorder[
                            ~df_reorder['Product Name'].isin(ordered_items)
                        ]
                        
                        # --- [MODIFIED] Display the filtered dataframe ---
                        st.dataframe(
                            df_reorder_to_display, # <-- CHANGED
                            use_container_width=True,
                            column_config={
                                "Product Name": st.column_config.TextColumn("Product Name (from Sales)", width="large"),
                                "Barcode": st.column_config.TextColumn("Barcode"),
                                "Stock": st.column_config.NumberColumn("Current Stock", format="%.0f units"),
                                "suggested_stock_level": st.column_config.NumberColumn("Suggested Stock", format="%.0f units"),
                                "deficit": st.column_config.NumberColumn("Need to Order", format="%.0f units"),
                                "avg_weekly_sales": st.column_config.NumberColumn("Avg. Weekly Sales", format="%.2f"),
                                "total_quantity_sold": st.column_config.NumberColumn("Total Sold"),
                            }
                        )
                        
                        # --- [NEW] Show a list of items you've marked as ordered ---
                        if ordered_items:
                            st.subheader("Already Ordered (Hidden from list above)")
                            st.dataframe(
                                df_reorder[df_reorder['Product Name'].isin(ordered_items)],
                                use_container_width=True,
                                column_config={
                                    "Product Name": st.column_config.TextColumn("Product Name (from Sales)", width="large"),
                                    "Barcode": st.column_config.TextColumn("Barcode"),
                                    "Stock": st.column_config.NumberColumn("Current Stock", format="%.0f units"),
                                    "suggested_stock_level": st.column_config.NumberColumn("Suggested Stock", format="%.0f units"),
                                    "deficit": st.column_config.NumberColumn("Need to Order", format="%.0f units"),
                                    "avg_weekly_sales": st.column_config.NumberColumn("Avg. Weekly Sales", format="%.2f"),
                                    "total_quantity_sold": st.column_config.NumberColumn("Total Sold"),
                                }
                            )
                        
                        with st.expander("View Full Stock-Sales Comparison (All Non-UTC Items)"):
                            st.dataframe(df_merged.sort_values(by='deficit', ascending=False), use_container_width=True)
        
        # Add a warning in the sidebar if stock file is missing
        elif uploaded_file is not None or utc_file is not None: # Only show if sales is loaded but stock isn't
            st.sidebar.warning("Upload your stock file to enable the 'Reorder & Stock Check' tab.")

# --- [FIXED] Removed the extra '}' ---
