public class DecisionTreeNode {

    DecisionTreeNode left, right;
    String column, value, label;

    public boolean isLeaf() {
        return left == null && right == null;
    }

    public void prettyPrint(StringBuilder buffer, String parent) {
        System.out.print(buffer);
        System.out.println("|");
        System.out.print(buffer);

        if (isLeaf()) {
            System.out.println("+" + parent + "->'class'=" + label);
        } else {
            System.out.println("+" + parent + "->Is " + column + " == " + value + "?");
            buffer.append("   |");
            left.prettyPrint(buffer, "Yes");
            buffer.deleteCharAt(buffer.length() - 1);
            buffer.append(" ");
            right.prettyPrint(buffer, "No");
            buffer.delete(buffer.length() - 4, buffer.length());
        }
    }

}
