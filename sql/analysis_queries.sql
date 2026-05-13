

-- Average Exam Score
SELECT AVG(Exam_Score)
FROM student_behavior;

-- Burnout Analysis
SELECT
    Burnout_Risk,
    AVG(Exam_Score)
FROM student_behavior
GROUP BY Burnout_Risk;

