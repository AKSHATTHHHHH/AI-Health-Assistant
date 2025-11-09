from flask import Flask, request, jsonify
import os
import tempfile
import mysql.connector

app = Flask(__name__)

def get_connection():
    # Save CA cert content to temp file
    ca_cert_content = os.environ["127.0.0.1"]
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as ca_file:
        ca_file.write(ca_cert_content)
        ca_path = ca_file.name

    conn = mysql.connector.connect(
        host=os.environ["localhost"],
        user=os.environ["root"],
        password=os.environ["Freefire@113"],
        port=int(os.environ["3306"]),
        database=os.environ["ai_health_db"],
        ssl_ca=ca_path,
        ssl_disabled=False
    )
    return conn


@app.route('/submit', methods=['POST'])
def submit_data():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "Invalid or missing JSON"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()

        values = (
            data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
            data['fbs'], data['restecg'], data['thalach'], data['exang'],
            data['oldpeak'], data['slope'], data['ca'], data['thal'],
            data['prediction'], data['confidence'], data['rule_diagnosis']
        )

        query = """
        INSERT INTO patient_data (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                                  oldpeak, slope, ca, thal, prediction, confidence, rule_diagnosis)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"message": "Data inserted successfully"}), 200
    except KeyError as e:
        return jsonify({"error": f"Missing key: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
