import streamlit as st
import pandas as pd
import plotly.express as px
import io
from prophet import Prophet
from prophet.plot import plot_plotly

# --- Page Configuration ---
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Helper Function to Load Data ---
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
            return None, None
            
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
            return None, None
            
        # --- [DEBUG] Print raw dates and types ---
        print("--- [DEBUG] Raw 'Sales Date' and 'Sales Time' (head) ---")
        print(df[['Sales Date', 'Sales Time']].head())
        print("\n--- [DEBUG] Data types of 'Sales Date' and 'Sales Time' ---")
        print(df[['Sales Date', 'Sales Time']].dtypes)
        # --- End Debug ---

        # Convert time to string just in case it's read as a time object
        df['Sales Time'] = df['Sales Time'].astype(str)
        
        df['datetime'] = pd.to_datetime(
            df['Sales Date'].astype(str).str.split(' ').str[0] + ' ' + df['Sales Time'], 
            errors='coerce',
            # dayfirst=True  # <-- REMOVED. This was the cause of the parsing error.
                             # Pandas will now correctly auto-detect YYYY-MM-DD
        )
        
        # --- [DEBUG] Print parsed dates ---
        print("\n--- [DEBUG] 'datetime' column after parsing (head) ---")
        print(df['datetime'].head())
        print(f"--- [DEBUG] Is 'datetime' column all NaT? {df['datetime'].isna().all()} ---")
        # --- End Debug ---
        
        # 2. Drop rows where date/time conversion failed
        df.dropna(subset=['datetime'], inplace=True)
        
        if df.empty:
            print("--- [DEBUG] DataFrame is EMPTY after dropping NaT datetimes. All date parsing failed. ---")
            st.error("Error: All date parsing failed. Please check your source file's date format.")
            return None, 1 # Return 1 for num_days to avoid crash
        else:
            print(f"--- [DEBUG] Min Date after dropna: {df['datetime'].min()}, Max Date after dropna: {df['datetime'].max()} ---")
        
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
        
        df.dropna(subset=['Total Profit', 'Quantity Sold', 'Invoice No'], inplace=True)
        
        # --- Get date range for inventory calculations ---
        
        # **THE FIX**: Calculate the number of *unique days* with sales,
        # not the total time span. This is robust to outlier dates.
        num_days = df['datetime'].dt.date.nunique()
        
        if num_days == 0:
            num_days = 1 # Avoid division by zero if only one day of data

        # Add a message to the sidebar to debug the date range
        st.sidebar.info(f"Date range found: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        st.sidebar.success(f"Calculating velocity based on **{num_days}** unique sales day(s).")
        print(f"--- [DEBUG] num_days calculated as: {num_days} ---")
        
        return df, num_days

    except Exception as e:
        st.error(f"Error loading sales data: {e}")
        st.error("Please ensure 'openpyxl' is installed (`pip install openpyxl`) and the file is a valid .xlsx file.")
        return None, None
        
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
        # The script needs 'Barcode', 'Stock', 'Product Name'.
        # Your provided list contains these exact names, so no rename map is needed.
        if 'Barcode' not in df.columns or 'Stock' not in df.columns or 'Product Name' not in df.columns:
            st.error("Stock file is missing 'Barcode', 'Stock', or 'Product Name' after renaming. Please check the column order.")
            return None
        
        # --- Select and clean required columns ---
        df_stock = df[['Barcode', 'Stock', 'Product Name']].copy()
        df_stock['Stock'] = pd.to_numeric(df_stock['Stock'], errors='coerce')

        # --- FIX: Convert Barcode from float to int to string ---
        # 1. Convert to string and strip whitespace
        df_stock['Barcode'] = df_stock['Barcode'].astype(str).str.strip()
        
        # 2. Convert to numeric to standardize (handles floats, ints, and strings-of-ints)
        df_stock['Barcode'] = pd.to_numeric(df_stock['Barcode'], errors='coerce')

        # 3. Drop rows where barcode/stock couldn't be converted
        df_stock.dropna(subset=['Stock', 'Barcode'], inplace=True)

        # 4. Convert to integer (to remove '.0') then to string
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

# --- Main App ---
st.title("🛒 Sales & Inventory Dashboard")

# --- Sidebar for File Upload ---
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("1. Upload Sales Data (.xlsx)", type=["xlsx"])
stock_file = st.sidebar.file_uploader("2. Upload Stock Data (.xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.info("Please upload your sales data file using the sidebar to get started.")
else:
    # --- Load and Process Data ---
    df, num_days = load_data(uploaded_file)
    
    if df is not None:
        
        # --- Sidebar Filters ---
        st.sidebar.header("Filters")
        
        # Filter by Product
        all_products = df['Product Name'].unique()
        selected_products = st.sidebar.multiselect(
            "Filter by Product",
            options=all_products,
            default=all_products
        )
        
        # Filter data based on sidebar selection
        df_filtered = df[df['Product Name'].isin(selected_products)]
        
        if df_filtered.empty:
            st.warning("No data found for the selected filters.")
        else:
            # --- Create Tabs ---
            tab_list = [
                "🔥 Busiest Times Heatmap", 
                "🏆 Product Performance", 
                "🛒 Customer Purchase Insights", # <-- ADDED
                "📦 Inventory Insights",
                "📈 Overall Trend & Forecasting"
            ]
            
            # Dynamically add the new tab if stock file is present
            if stock_file is not None:
                tab_list.append("🛍️ Reorder & Stock Check")
            
            tabs = st.tabs(tab_list)

            # --- Tab 1: Busiest Times Heatmap ---
            with tabs[0]:
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

            # --- Tab 2: Product Performance ---
            with tabs[1]:
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
                best_product = sorted_products.iloc[0]
                col1.metric(
                    f"Best Product (by {sort_by})",
                    best_product['Product Name'],
                )
                col2.metric(
                    f"Value (by {sort_by})",
                    f"฿{best_product[chart_col]:,.2f}" if chart_col != 'total_quantity_sold' else f"{best_product[chart_col]:,.0f} Units"
                )

                # --- Bar Chart of Top 10 ---
                st.subheader("Top 10 Products")
                top_10_products = sorted_products.head(10)
                
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

            # --- Tab 3: Customer Purchase Insights ---
            with tabs[2]:
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

            # --- Tab 4: Inventory Insights ---
            with tabs[3]:
                st.header("Inventory Insights & Sales Velocity")
                st.markdown(f"Calculations are based on the total sales over **{num_days}** days of data provided.")
                
                # --- MODIFICATION: Group by Barcode as well for merging ---
                if 'Product Barcode' not in df_filtered.columns:
                    st.error("Sales data is missing 'Product Barcode' column, which is needed to link to stock data.")
                    # Fallback to just product name if barcode isn't there
                    group_cols = ['Product Name']
                else:
                    group_cols = ['Product Name', 'Product Barcode']

                inventory_stats = df_filtered.groupby(group_cols).agg(
                    total_quantity_sold=('Quantity Sold', 'sum'),
                ).reset_index()
                
                # --- Rename 'Product Barcode' to 'Barcode' for merging ---
                if 'Product Barcode' in inventory_stats.columns:
                    inventory_stats.rename(columns={'Product Barcode': 'Barcode'}, inplace=True)
                    
                    # --- FIX: Robust Barcode Cleaning ---
                    # 1. Convert to string and strip whitespace
                    inventory_stats['Barcode'] = inventory_stats['Barcode'].astype(str).str.strip()
                    
                    # 2. Convert to numeric to standardize
                    inventory_stats['Barcode'] = pd.to_numeric(inventory_stats['Barcode'], errors='coerce')
                    
                    # 3. Drop NaNs created by conversion
                    inventory_stats.dropna(subset=['Barcode'], inplace=True)
                    
                    # 4. Convert to integer (to remove '.0') then to string
                    inventory_stats['Barcode'] = inventory_stats['Barcode'].astype('Int64').astype(str)

                # Calculate sales velocity
                inventory_stats['avg_daily_sales'] = inventory_stats['total_quantity_sold'] / num_days
                inventory_stats['avg_weekly_sales'] = inventory_stats['avg_daily_sales'] * 7
                
                # Add stocking suggestion
                st.subheader("Stocking Suggestions")
                lead_time_weeks = st.number_input("How many weeks of stock do you want to keep on hand? (Safety Stock)", min_value=1, max_value=12, value=2, step=1)
                
                inventory_stats['suggested_stock_level'] = inventory_stats['avg_weekly_sales'] * lead_time_weeks
                
                # Format for display
                inventory_display = inventory_stats.round(2)
                
                st.dataframe(
                    inventory_display.sort_values(by='avg_weekly_sales', ascending=False),
                    use_container_width=True,
                    column_config={
                        "Product Name": st.column_config.TextColumn("Product Name", width="large"),
                        "total_quantity_sold": st.column_config.NumberColumn("Total Sold", format="%d units"),
                        "avg_daily_sales": st.column_config.NumberColumn("Avg. Daily Sales"),
                        "avg_weekly_sales": st.column_config.NumberColumn("Avg. Weekly Sales"),
                        "suggested_stock_level": st.column_config.NumberColumn(f"Suggested {lead_time_weeks}-Week Stock", format="%.1f units"),
                    }
                )
            
            # --- Tab 5: Overall Trend & Forecasting ---
            with tabs[4]:
                st.header("Overall Sales Trend & Forecasting")
                
                # --- 1. Prepare data for Trend Plot ---
                st.subheader("Overall Daily Sales Trend")
                df_daily_sales = df_filtered.groupby(df_filtered['datetime'].dt.date) \
                                          .agg(daily_sales=('Total Price After Discount', 'sum')) \
                                          .reset_index()
                df_daily_sales = df_daily_sales.rename(columns={'datetime': 'date'})

                # Plot overall trend
                fig_trend = px.line(
                    df_daily_sales,
                    x='date',
                    y='daily_sales',
                    title="Total Daily Sales Over Time",
                    labels={'date': 'Date', 'daily_sales': 'Total Sales (฿)'}
                )
                fig_trend.update_layout(xaxis_title="Date", yaxis_title="Total Sales (฿)")
                st.plotly_chart(fig_trend, use_container_width=True)

                # --- 2. Prophet Forecasting ---
                st.subheader("Sales Forecast with Prophet")
                
                # Format for Prophet: needs 'ds' and 'y'
                df_prophet = df_daily_sales.rename(columns={'date': 'ds', 'daily_sales': 'y'})
                
                # Check for sufficient data
                if len(df_prophet) < 5:
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

                    # Run forecast
                    with st.spinner(f"Generating {forecast_days}-day forecast..."):
                        try:
                            m, forecast = get_prophet_forecast(df_prophet, forecast_days)
                            
                            st.subheader(f"{forecast_days}-Day Sales Forecast")
                            fig_forecast = plot_plotly(m, forecast)
                            fig_forecast.update_layout(
                                title=f"Daily Sales Forecast",
                                xaxis_title="Date",
                                yaxis_title="Forecasted Sales (฿)"
                            )
                            st.plotly_chart(fig_forecast, use_container_width=True)
                            
                            st.subheader("Forecast Components")
                            # Prophet's component plot uses matplotlib, so we use st.pyplot
                            fig_components = m.plot_components(forecast)
                            st.pyplot(fig_components)
                        
                        except Exception as e:
                            st.error(f"An error occurred during forecasting: {e}")
                            st.error("This can happen if there isn't enough varied data (e.g., all sales on one day).")
            
            # --- Tab 6: Reorder & Stock Check (Conditional) ---
            if stock_file is not None:
                with tabs[5]:
                    st.header("Stock vs. Forecasted Demand")
                    
                    if 'Barcode' not in inventory_stats.columns:
                        st.error("Cannot create reorder list. The sales data is missing the 'Product Barcode' column.")
                    else:
                        # Load stock data
                        df_stock = load_stock_data(stock_file)
                        
                        # We must check if df_stock is valid *before* we try to use it
                        if df_stock is not None:
                            # We are using the print statements you added in the prompt
                            print(df_stock.head()) 
                            
                            # Merge sales velocity (inventory_stats) with current stock (df_stock)
                            # We use a 'left' merge to start from what we've sold.
                            df_merged = pd.merge(
                                inventory_stats, # From sales data
                                df_stock[['Barcode', 'Stock']], # From stock data
                                on='Barcode',
                                how='left' 
                            )
                            print(df_merged.head())
                            
                            # Calculate the deficit
                            # Fillna(0) for items sold but not in stock file
                            df_merged['Stock'] = df_merged['Stock'].fillna(0)
                            df_merged['deficit'] = df_merged['suggested_stock_level'] - df_merged['Stock']
                            
                            st.subheader("Reorder List")
                            st.markdown(f"This list shows items where your `Current Stock` is *less* than the `Suggested Stock Level`. (Suggested level is based on **{lead_time_weeks} weeks** of safety stock, set in Tab 4).")
                            
                            # Filter for items that need reordering
                            df_reorder = df_merged[df_merged['deficit'] > 0].sort_values(by='deficit', ascending=False)
                            
                            st.dataframe(
                                df_reorder,
                                use_container_width=True,
                                column_config={
                                    "Product Name": st.column_config.TextColumn("Product Name (from Sales)", width="large"),
                                    "Barcode": st.column_config.TextColumn("Barcode"),
                                    "Stock": st.column_config.NumberColumn("Current Stock", format="%.0f units"),
                                    "suggested_stock_level": st.column_config.NumberColumn("Suggested Stock", format="%.1f units"),
                                    "deficit": st.column_config.NumberColumn("Need to Order", format="%.1f units"),
                                    "avg_weekly_sales": st.column_config.NumberColumn("Avg. Weekly Sales"),
                                    "total_quantity_sold": st.column_config.NumberColumn("Total Sold"),
                                }
                            )
                            
                            with st.expander("View Full Stock-Sales Comparison (All Items)"):
                                st.dataframe(df_merged.sort_values(by='deficit', ascending=False), use_container_width=True)
            
            # Add a warning in the sidebar if stock file is missing
            elif uploaded_file is not None: # Only show if sales is loaded but stock isn't
                st.sidebar.warning("Upload your stock file to enable the 'Reorder & Stock Check' tab.")
