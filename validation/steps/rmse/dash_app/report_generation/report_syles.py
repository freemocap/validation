styles = """
        <style>
            .marker-section {
                border-top: 2px solid #ccc;
                padding-top: 20px;
                margin-top: 20px;
            }
            h1 {
                page-break-before: always; /* Ensures each marker starts on a new page when printed */
            }
            .navigation {
                position: fixed;
                top: 0;
                left: 0;
                background-color: #f8f8f8;
                width: 100%;
                text-align: left;
                padding: 8px 0;
                border-bottom: 1px solid #e7e7e7;
                z-index: 1000;
            }
            .navigation a {
                padding: 6px 12px;
                text-decoration: none;
                color: #007bff;
            }
            .navigation a:hover {
                background-color: #e7e7e7;
            }
            .indicators-container {
                display: flex;
                flex-direction: row;
                align-items: center; /* Center align items horizontally in the column */
                justify-content: center; /* Center align items vertically in the column */
                margin-bottom: 5px; /* Space at the bottom of the container */
            }
            .indicator-wrapper {
                margin: 10px 0; /* Margin around each indicator for spacing */
                width: 15%; /* Full width of the container */
                height: 250px; 
                overflow: hidden; /* Hide content that exceeds the div height */
            }
            .rmse-tables-container {
                display: flex;
                flex-direction: row;
                justify-content: space-evenly;
                align-items: center;
            }

            table {
                    border-collapse: collapse;
                    width: 200px; /* Set the width of the table */
                    background-color: #f8f8f8; /* Background color */
                    border: 1px solid #ccc; /* Border color */
                    margin-bottom: 10px; /* Space after the table */
                }

            td {
                    padding: 5px; /* Padding inside each cell */
                    text-align: left; /* Align text to the left */
                }

            tr:nth-child(even) {background-color: #f2f2f2;} /* Zebra striping for rows */

            th {
                background-color: #f8f8f8; /* Header background color */
                border-bottom: 1px solid #ccc; /* Border color */
                padding: 5px; /* Padding inside each cell */
                text-align: left; /* Align text to the left */
            }
            }


        </style>
"""